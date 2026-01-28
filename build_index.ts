import { VectorStoreIndex, storageContextFromDefaults, Settings, Document } from "llamaindex";
import { PDFReader } from "@llamaindex/readers/pdf";
import { HuggingFaceEmbedding } from "@llamaindex/huggingface";
import fs from "fs/promises";
import path from "path";

Settings.embedModel = new HuggingFaceEmbedding({
    modelType: "sentence-transformers/all-MiniLM-L6-v2"
});

async function runIndexer() {
    const pdfFolder = "./pdf";
    const pdfReader = new PDFReader();
    let allPages: Document[] = [];

    console.log("📂 Scanning PDF folder...");
    const files = await fs.readdir(pdfFolder);
    const pdfFiles = files.filter(f => f.endsWith(".pdf"));

    for (const file of pdfFiles) {
        console.log(`📄 Processing: ${file}`);
        const filePath = path.join(pdfFolder, file);
        
        // MANUALLY load each PDF. This returns an array of Documents (one per page).
        const pages = await pdfReader.loadData(filePath);
        
        // Ensure metadata is correctly attached to every single page
        pages.forEach((page, index) => {
            page.metadata = {
                ...page.metadata,
                file_name: file,
                page_label: (index + 1).toString() // Manual fallback for page number
            };
            // Prevent LLM from seeing the metadata text, but keep it in the index
            page.excludedLlmMetadataKeys = ["file_size", "creation_date"];
            
            console.log(`   ✅ Indexed Page ${page.metadata.page_label}`);
        });

        allPages = allPages.concat(pages);
    }

    console.log(`\n🧠 Total chunks to index: ${allPages.length}`);
    const storageContext = await storageContextFromDefaults({ persistDir: "./storage" });
    await VectorStoreIndex.fromDocuments(allPages, { storageContext });

    console.log("✅ Local Database built successfully in ./storage");
}

runIndexer().catch(console.error);
