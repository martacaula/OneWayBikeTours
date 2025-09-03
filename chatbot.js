// chatbot.js
import 'dotenv/config'; // loads .env automatically
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://models.github.ai/inference",
  apiKey: process.env.GITHUB_TOKEN,
});

async function main() {
  const messages = [
    { role: "system", content: "You are a helpful chatbot." }
  ];

  console.log("Chatbot ready. Type 'exit' to quit.\n");

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
          temperature: 0.7,
        });

        const reply = response.choices[0].message.content;
        console.log("Bot:", reply, "\n");

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
