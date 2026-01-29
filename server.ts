import express, {Request, Response} from 'express';
import { queryData } from './chat.js';

const app = express();
const PORT = 3000;

app.use(express.json());

app.get('/health', (req, res) => {
    res.status(200).send('OK');
});

app.post ('/ask', async(req: Request, res:Response) => {
    const { question } = req.body;
    if (!question) {
        return res.status(400).json({ error: "No question provided" });
    }

    try {
        console.log(`Received question: ${question}`);
        const result = await queryData(question);
        res.json(result);
    }
    catch (error) {
        console.error("API Error: ", error);
        res.status(500).json({ error: "Internal server error" });
    }
});


app.listen(PORT, () => {
    console.log(`RAG Worder running on http://localhost:${PORT}`);
})
