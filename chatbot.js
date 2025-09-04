// chatbot.js
import 'dotenv/config'; // loads .env automatically
import OpenAI from "openai";
import fs from 'node:fs/promises';
import path from 'node:path';

const client = new OpenAI({
  baseURL: "https://models.github.ai/inference",
  apiKey: process.env.GITHUB_TOKEN,
});

// -----------------------------
// RAG: Context loading & retrieval
// -----------------------------
const CONTEXT_PATH = path.join(process.cwd(), 'context', 'context.md');
const EMBEDDING_MODEL = 'openai/text-embedding-3-small';
const TOP_K = 5;

let contextChunks = [];
let chunkEmbeddings = [];

function normalize(vec) {
  const norm = Math.sqrt(vec.reduce((acc, v) => acc + v * v, 0)) || 1;
  return vec.map(v => v / norm);
}

function cosineSim(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length && i < b.length; i++) sum += a[i] * b[i];
  return sum;
}

function chunkTextByParagraphs(text, maxChars = 800) {
  const paras = text
    .split(/\n\s*\n/g)
    .map(p => p.trim())
    .filter(Boolean);
  const chunks = [];
  let buf = '';
  for (const p of paras) {
    if ((buf + '\n\n' + p).trim().length <= maxChars) {
      buf = (buf ? buf + '\n\n' : '') + p;
    } else {
      if (buf) chunks.push(buf);
      if (p.length <= maxChars) {
        buf = p;
      } else {
        // hard wrap very long paragraphs
        for (let i = 0; i < p.length; i += maxChars) {
          chunks.push(p.slice(i, i + maxChars));
        }
        buf = '';
      }
    }
  }
  if (buf) chunks.push(buf);
  return chunks;
}

async function ensureContextEmbeddings() {
  if (chunkEmbeddings.length && contextChunks.length) return; // already prepared
  try {
    const raw = await fs.readFile(CONTEXT_PATH, 'utf-8');
    contextChunks = chunkTextByParagraphs(raw);
    if (contextChunks.length === 0) return;
    const embResp = await client.embeddings.create({
      model: EMBEDDING_MODEL,
      input: contextChunks,
    });
    chunkEmbeddings = embResp.data.map(d => normalize(d.embedding));
  } catch (e) {
    console.warn('RAG initialization warning:', e?.message || e);
    contextChunks = [];
    chunkEmbeddings = [];
  }
}

async function retrieveContext(query, k = TOP_K) {
  if (!contextChunks.length || !chunkEmbeddings.length) return [];
  try {
    const qEmbResp = await client.embeddings.create({
      model: EMBEDDING_MODEL,
      input: query,
    });
    const qVec = normalize(qEmbResp.data[0].embedding);
    const scored = chunkEmbeddings.map((vec, idx) => ({ idx, score: cosineSim(vec, qVec) }));
    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, Math.min(k, scored.length));
    return top.map(({ idx, score }) => ({ score, text: contextChunks[idx] }));
  } catch (e) {
    console.warn('RAG retrieval warning:', e?.message || e);
    return [];
  }
}

function buildRagSystemMessage(topSnippets) {
  if (!topSnippets.length) return null;
  const contextBlock = topSnippets.map((s, i) => `[[Snippet ${i + 1} (score=${s.score.toFixed(3)})]]\n${s.text}`).join('\n\n---\n\n');
  const instruction = [
    'You are a helpful assistant for One Way Bike Tours. Use ONLY the information in the CONTEXT below to answer. If the answer is not in the context, say you do not know and ask for clarification. Keep answers brief, clear, and aligned with company policies. If the issue remains unresolved, offer a phone call.',
    '',
    'CONTEXT:',
    contextBlock,
  ].join('\n');
  return { role: 'system', content: instruction };
}

async function main() {
  const messages = [
    { role: "system", content: "You are the CEO of a company, responsible for answering clients' questions about the products or services your company provides." }
  ];

  // Prepare RAG context (non-blocking message shown once)
  await ensureContextEmbeddings();
  if (contextChunks.length) {
    console.log(`[RAG] Loaded ${contextChunks.length} context chunks from ${CONTEXT_PATH}`);
  } else {
    console.log(`[RAG] No context available or failed to load context. Continuing without augmentation.`);
  }

  console.log("Hi! How can I help you today?\n"); // Type 'exit' to quit.

  // use readline for interactive CLI
  const readline = await import("readline");
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  async function ask() {
    rl.question("You: ", async (userInput) => {
      if (userInput.toLowerCase() === "exit") {
        rl.close();
        return;
      }

      messages.push({ role: "user", content: userInput });

      try {
        // Retrieve RAG snippets for this query
        const topSnippets = await retrieveContext(userInput, TOP_K);
        const ragSystem = buildRagSystemMessage(topSnippets);

        // Build augmented message list for this turn
        const augmentedMessages = ragSystem ? [...messages, ragSystem] : [...messages];

        const response = await client.chat.completions.create({
          model: "openai/gpt-4o",   // or another supported model
          messages: augmentedMessages,
          temperature: 0.5,
        });

        const reply = response.choices[0].message.content;
        console.log("Gerrit:", reply, "\n");

        messages.push({ role: "assistant", content: reply });
      } catch (err) {
        console.error("Error:", err.message);
      }

      ask(); // loop back
    });
  }

  ask();
}

main();
