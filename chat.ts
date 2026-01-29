import { createOpenAI } from '@ai-sdk/openai';
import { generateObject } from 'ai';
import { z } from 'zod';
import { VectorStoreIndex, storageContextFromDefaults, Settings } from "llamaindex";
import { HuggingFaceEmbedding } from "@llamaindex/huggingface"

// 1. Same Embedding model as the indexer
Settings.embedModel = new HuggingFaceEmbedding({
    modelType: "sentence-transformers/all-MiniLM-L6-v2"
});

// 2. Point the OpenAI SDK to your local Ollama
const ollama = createOpenAI({
    baseURL: 'http://localhost:11434/v1',
    apiKey: 'ollama', 
});

export async function queryData(question: string) {
    console.log("🔍 Searching local index...");
    
    const storageContext = await storageContextFromDefaults({ persistDir: "./storage" });
    const index = await VectorStoreIndex.init({ storageContext });
    
    const retriever = index.asRetriever({ similarityTopK: 3 });
    const matches = await retriever.retrieve({ query: question });

    // 1. EXTRACT AND SORT SOURCES
    const sourceObjects = matches.map(m => ({
        file: m.node.metadata.file_name || "Unknown",
        page: m.node.metadata.page_label || m.node.metadata.page_number || "0"
    }));

    // Perform the sort: Filename (A-Z) then Page Number (Numeric)
    sourceObjects.sort((a, b) => {
        // Sort by filename first
        const fileCompare = a.file.localeCompare(b.file, undefined, { sensitivity: 'base' });
        if (fileCompare !== 0) return fileCompare;

        // If same file, sort by page number numerically
        return parseInt(a.page) - parseInt(b.page);
    });

    // Format into clean strings and remove duplicates
    const sortedSources = [...new Set(sourceObjects.map(s => `${s.file} (Page ${s.page})`))];

    const contextText = matches.map(m => m.node.getContent()).join("\n\n");

    console.log("🤖 Generating response with Qwen...");
    
    const { object } = await generateObject({
        model: ollama('qwen2.5:3b'),
        schema: z.object({
            answer: z.string().describe("Direct answer based on the context.")
        }),
        prompt: `Context:\n${contextText}\n\nQuestion: ${question}`,
    });

    // 2. RETURN COMBINED OBJECT
    return {
        ...object,
        sources: sortedSources
    };
}

// Replace this with a question about your PDFs
queryData("What are the changes made to VW 60361?").then(console.log).catch(console.error);

