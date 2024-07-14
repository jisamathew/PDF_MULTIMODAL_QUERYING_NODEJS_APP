import dotenv from 'dotenv';
import express from 'express';
import bodyParser from "body-parser";
import { HfInference } from '@huggingface/inference';
import fetch from 'node-fetch';
import { MongoClient } from 'mongodb';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.json()); // Add this to parse JSON bodies

dotenv.config();

// const app = express();
// app.set('view engine', 'ejs');
// app.use(bodyParser.urlencoded({ extended: true }));
// app.use(express.json());

console.log('HUGGING_FACE_API_TOKEN:', process.env.HUGGING_FACE_API_TOKEN);
console.log('MONGO_URI:', process.env.MONGO_URI);
console.log('PORT:', process.env.PORT);
// const headers =   `Bearer ${process.env.HUGGING_FACE_API_TOKEN}`
const inference = new HfInference( `${process.env.HUGGING_FACE_API_TOKEN}`, { fetch: fetch });
const uri = process.env.MONGO_URI;
const client = new MongoClient(uri);

// Middleware to ensure MongoDB client is connected
app.use(async (req, res, next) => {
    console.log('Middleware execution started');
    try {
        const db = client.db('sample_hscode'); // Adjust as needed for different databases
        const topology = client.topology;
        
        if (topology && topology.isConnected()) {
            console.log('MongoDB already connected');
        } else {
            console.log('MongoDB not connected, connecting...');
            await client.connect();
            console.log('MongoDB connected');
        }
        
        req.db = db;
        next();
    } catch (error) {
        console.error('MongoDB connection error:', error);
        res.status(500).send('Internal server error');
    }
});

app.get("/", (req, res) => {
    res.sendFile(__dirname + "/index.html");
});


async function searchSimilarText(queryText) {
  console.log('Search Similar Text Function')
  console.log(queryText)
    // const inputs = {
    //   "inputs": {
    //     "text": queryText
    //   },
    // };
  
    const response = await inference.request({
      model: 'openai/clip-vit-base-patch32',
      inputs: queryText
      
    });
    // const inputs = {
    //     inputs: {
    //       text: queryText,
    //     },
    //   };
    //     const response = await inference.request({
    //       model: 'sentence-transformers/all-MiniLM-L6-v2',
    //       inputs: [queryText],
    //     });
  console.log('Getting response')
  console.log(response)
    const queryEmbedding = response.text_embeddings[0];
  
    const pipeline = [
      {
        "$vectorSearch": {
          "index": "PDFIndex",
          "path": "embedding",
          "queryVector": queryEmbedding,
          "numCandidates": 100,
          "limit": 20,
        }
      },
      {
        "$project": {
          "_id": 1,
          "score": {"$meta": "vectorSearchScore"},
          "source": 1,
          "type": 1,
          "image_path": 1,
          "text": 1,
          "table": 1,
          "metadata": 1
        }
      }
    ];
  
    const collection = client.db('sample_hscode').collection('pdf_multimodal_embedding_version3');
    console.log('collection')
    console.log(collection)
    const results = await collection.aggregate(pipeline).toArray();
    console.log('result')
    console.log(results)
    // Group results by source
    const groupedResults = {};
    results.forEach(result => {
      const source = result.source;
      if (!groupedResults[source]) {
        groupedResults[source] = [];
      }
      groupedResults[source].push(result);
    });
  
    return groupedResults;
  }
  
  app.post('/search', async (req, res) => {
    const queryText = req.body.searchInput;
    
    console.log('queryText')
    console.log(queryText)
    try {
      await client.connect();
      console.log('going to search similar text')
      const groupedResults = await searchSimilarText(queryText);
      res.json(groupedResults);
    } catch (error) {
      console.error('Error during search:', error);
      res.status(500).send('Internal Server Error');
    } finally {
      await client.close();
    }
  });

app.listen(process.env.PORT, function () {
    console.log("server is running on port " + process.env.PORT);
});

// Call createEmbeddings function to create embeddings
// createEmbeddings();

// {
//     "name": "my-app",
//     "version": "1.0.0",
//     "description": "",
//     "main": "server.mjs",
//     "type": "module",
//     "scripts": {
//       "test": "echo \"Error: no test specified\" && exit 1"
//     },
//     "author": "",
//     "license": "ISC",
//     "dependencies": {
//       "@huggingface/inference": "^2.7.0",
//       "axios": "^1.6.8",
//       "body-parser": "^1.20.0",
//       "dotenv": "^16.0.0",
//       "ejs": "^3.1.6",
//       "express": "^4.17.3",
//       "mongoose": "^6.2.10",
//       "node-fetch": "^3.3.2"
//     }
//   }
  