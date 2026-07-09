import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import helmet from "helmet";
import rateLimit from "express-rate-limit";

async function startServer() {
  // Strip --port and --host from process.argv to prevent Vite from crashing
  process.argv = process.argv.filter(arg => !arg.startsWith('--port') && !arg.startsWith('--host'));

  const app = express();
  const PORT = 3000;

  app.set("trust proxy", 1); // Trust first proxy

  // Security headers (CSP disabled to avoid breaking Vite's inline scripts/styles)
  app.use(helmet({ contentSecurityPolicy: false }));

  // Rate limiting for the API endpoint (e.g., max 20 requests per minute per IP)
  const apiLimiter = rateLimit({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 20,
    message: { error: "Too many requests from this IP, please try again later." },
    standardHeaders: true,
    legacyHeaders: false,
    validate: { xForwardedForHeader: false, trustProxy: false }
  });

  app.use(express.json({ limit: "20mb" })); // Reduced limit to a more reasonable size

  app.post("/api/analyze-audio", apiLimiter, async (req, res) => {
    try {
      const { audioBase64, mimeType } = req.body;
      
      if (!audioBase64 || typeof audioBase64 !== 'string') {
        return res.status(400).json({ error: "Invalid or missing audio data" });
      }

      // Basic length validation (e.g., limit to ~15MB base64 string)
      if (audioBase64.length > 15 * 1024 * 1024) {
        return res.status(413).json({ error: "Audio payload too large" });
      }

      if (mimeType && typeof mimeType !== 'string') {
        return res.status(400).json({ error: "Invalid mime type" });
      }

      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        console.error("Missing GEMINI_API_KEY environment variable");
        return res.status(500).json({ error: "Internal server configuration error." });
      }

      const ai = new GoogleGenAI({ apiKey });
      
      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: [
          {
             role: 'user',
             parts: [
               {
                 inlineData: {
                   data: audioBase64,
                   mimeType: mimeType || "audio/webm"
                 }
               },
               {
                 text: "Listen to this audio. Provide the transcription of what is being said, and identify the language (and country if applicable). Format the response as a JSON object with keys 'transcription' and 'language'. If no language can be identified or there is no speech, leave transcription as empty and language as '알 수 없음'. Write the language in Korean (e.g. '영어', '한국어', '스페인어')."
               }
             ]
          }
        ],
        config: {
          responseMimeType: "application/json",
          temperature: 0.1,
        }
      });

      const resultText = response.text?.trim() || "{}";
      const result = JSON.parse(resultText);

      res.json(result);
    } catch (error: any) {
      console.error("Error analyzing audio:", error);
      // Avoid leaking internal error messages to the client
      res.status(500).json({ error: "Failed to analyze audio due to an internal server error." });
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true, hmr: false },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
