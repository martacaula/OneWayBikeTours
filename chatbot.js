// chatbot.js
import 'dotenv/config'; // loads .env automatically
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://models.github.ai/inference",
  apiKey: process.env.GITHUB_TOKEN,
});

async function main() {
  const messages = [
    { role: "system", content: "You arYou are the CEO of a company, responsible for answering clients' questions about the products or services your company provides.  If you are unsure of an answer, request clarification or additional information as needed. Ensure all responses are brief, clear, concise , and aligned with any specific instructions provided.If you are unable to resolve the issue after your response, ask clients if they would prefer to discuss the matter further via a direct phone call.e a helpful human." }
  ];

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
        const response = await client.chat.completions.create({
          model: "openai/gpt-4o",   // or another supported model
          messages,
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
