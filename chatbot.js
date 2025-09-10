// chatbot.js
import 'dotenv/config'; // loads .env automatically
import OpenAI from "openai";
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';

const client = new OpenAI({
  baseURL: "https://models.github.ai/inference",
  apiKey: process.env.GITHUB_TOKEN,
});

// -----------------------------
// RAG: Context loading & retrieval
// -----------------------------
const CONTEXT_PATH = path.join(process.cwd(), 'context', 'context.md');
const CACHE_PATH = path.join(process.cwd(), 'context', 'context.md.cache.json');
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

function sha256(text) {
  return crypto.createHash('sha256').update(text, 'utf8').digest('hex');
}

async function loadCache() {
  try {
    const raw = await fs.readFile(CACHE_PATH, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

async function saveCache(payload) {
  try {
    await fs.writeFile(CACHE_PATH, JSON.stringify(payload, null, 2), 'utf-8');
  } catch (e) {
    console.warn('[RAG] Failed to write cache:', e?.message || e);
  }
}

async function ensureContextEmbeddings() {
  if (chunkEmbeddings.length && contextChunks.length) return; // already prepared
  try {
    const raw = await fs.readFile(CONTEXT_PATH, 'utf-8');
    const fileHash = sha256(raw);

    // Try load cache
    const cache = await loadCache();
    if (cache && cache.fileHash === fileHash && cache.embeddingModel === EMBEDDING_MODEL) {
      contextChunks = cache.chunks || [];
      chunkEmbeddings = (cache.embeddings || []).map(normalize);
      if (contextChunks.length && chunkEmbeddings.length === contextChunks.length) {
        console.log(`[RAG] Cache hit: ${CACHE_PATH}`);
        return;
      }
    }

    // Recompute and cache
    contextChunks = chunkTextByParagraphs(raw);
    if (contextChunks.length === 0) return;
    const embResp = await client.embeddings.create({
      model: EMBEDDING_MODEL,
      input: contextChunks,
    });
    chunkEmbeddings = embResp.data.map(d => normalize(d.embedding));
    await saveCache({
      version: 1,
      embeddingModel: EMBEDDING_MODEL,
      fileHash,
      chunks: contextChunks,
      embeddings: embResp.data.map(d => d.embedding),
    });
  } catch (e) {
    console.warn('RAG initialization warning:', e?.message || e);
    contextChunks = [];
    chunkEmbeddings = [];
  }
}

const STOPWORDS = new Set([
  'the','and','for','are','but','with','that','this','from','into','over','your','you','our','can','will','have','has','was','were','not','all','any','about','how','what','when','where','why','which','who','whom','is','to','in','on','of','it','as','at','by','be','or','an','a'
]);

function keywordSet(text) {
  const tokens = (text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).filter(t => t.length > 2 && !STOPWORDS.has(t));
  return new Set(tokens);
}

async function retrieveContext(query, k = TOP_K) {
  if (!contextChunks.length || !chunkEmbeddings.length) return [];
  try {
    const qEmbResp = await client.embeddings.create({
      model: EMBEDDING_MODEL,
      input: query,
    });
    const qVec = normalize(qEmbResp.data[0].embedding);

    // Keyword prefiltering
    const kws = keywordSet(query);
    let candidateIdx = [];
    if (kws.size) {
      const kwArr = Array.from(kws);
      for (let i = 0; i < contextChunks.length; i++) {
        const lc = contextChunks[i].toLowerCase();
        if (kwArr.some(w => lc.includes(w))) candidateIdx.push(i);
      }
    }
    if (candidateIdx.length === 0) {
      candidateIdx = contextChunks.map((_, i) => i); // fallback to all
    }

    const scored = candidateIdx.map(idx => ({ idx, score: cosineSim(chunkEmbeddings[idx], qVec) }));
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
    'Provide brief, clear, and conversational responses. If a question is unclear, ask the client for clarification. If uncertain about an answer, offer a proactive solution offer to connect the client with the appropriate company representative for further assistance. When information is unavailable, focus on providing helpful next steps or solutions.are a helpful assistant for One Way Bike Tours. Use ONLY the information in the CONTEXT below to answer. If the answer is not in the context, ask for clarification. Keep answers brief, clear, and aligned with company policies. If the issue remains unresolved, offer a phone call.',
    '',
    'CONTEXT:',
    contextBlock,
  ].join('\n');
  return { role: 'system', content: instruction };
}

async function main() {
  const messages = [
    { role: "system", content: "Act as the CEO of a company, responding to client inquiries about your company's products and services. After responding, validate in 1-2 lines that your answer is clear, concise, and directly addresses the client's question or provides a next step. Self-correct briefly if not." }
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
