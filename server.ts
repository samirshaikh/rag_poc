import express from 'express';
import { createOpenAI } from '@ai-sdk/openai';
import { streamText, pipeUIMessageStreamToResponse } from 'ai';
import { 
    VectorStoreIndex, 
    storageContextFromDefaults, 
    Settings 
} from "llamaindex";
import { HuggingFaceEmbedding } from "@llamaindex/huggingface";
// 1. Setup Global Settings (Must match build_index.ts)
Settings.embedModel = new HuggingFaceEmbedding({
    modelType: "sentence-transformers/all-MiniLM-L6-v2"
});

import os from 'os';

const app = express();
app.use(express.json());

// Configure Ollama Provider (OpenAI Compatible)
const ollama = createOpenAI({
    baseURL: 'http://localhost:11434/v1',
    apiKey: 'ollama', 
});

async function getPdfContext(question: string) {
    const storageContext = await storageContextFromDefaults({ persistDir: "./storage" });
    const index = await VectorStoreIndex.init({ storageContext });
    const retriever = index.asRetriever({ similarityTopK: 3 });
    const nodes = await retriever.retrieve({ query: question });

    return nodes.map( n=> ({
        text: n.node.getContent(),
        file: n.node.metadata.file_name,
        page: n.node.metadata.page_label
    }))
}

app.get('/health', async(req, res) => {
    const hostname = os.hostname();
    console.log("Health check");
    res.status(200).json({ status: "OK", server: hostname, timestamp: new Date().toISOString() });
});

// The Main Chat Endpoint
//
app.post('/api/chat', async (req, res) => {
    const { messages } = req.body;
    const lastUserMessage = messages[messages.length - 1].content;
    
    console.log(`Querying LLM: ${os.hostname()}`);
    try {
        const contextNodes = await getPdfContext(lastUserMessage);
        const contextString = contextNodes
            .map(n => `[SOURCE: ${n.file} | PAGE: ${n.page}]\n${n.text}`)
            .join("\n\n");

        // 1. Set headers to DISABLE BUFFERING
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.setHeader('Transfer-Encoding', 'chunked');
        res.setHeader('X-Content-Type-Options', 'nosniff');
        res.setHeader('Cache-Control', 'no-cache, no-transform');

        const result = streamText({
            model: ollama('qwen2.5:3b'),
            system: `Use the context to answer.\n\nCONTEXT:\n${contextString}`,
            messages,
        });

        // 2. Use pipeTextStreamToResponse for the simplest raw output
        result.pipeTextStreamToResponse(res);

    } catch (error: any) {
        console.error("❌ Stream Error:", error.message);
        if (!res.headersSent) {
            res.status(500).json({ error: "Internal Server Error" });
        }
    }
});

const PORT = 3000;
app.use(express.static('public'));

app.listen(PORT, () => {
    console.log(`🚀 Server ready at http://localhost:${PORT}`);
    console.log(`📡 Endpoint: http://localhost:${PORT}/api/chat`);
});
