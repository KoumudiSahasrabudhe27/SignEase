import { SOURCE_LANGUAGE, type TargetLanguageCode } from "@/constants/languages";

const TRANSLATION_API_BASE =
  import.meta.env.VITE_TRANSLATION_API_URL ?? "http://localhost:8002";

export interface TranslateRequest {
  text: string;
  source_lang?: string;
  target_lang: TargetLanguageCode;
}

export interface TranslateResponse {
  translation: string;
  source_lang: string;
  target_lang: string;
}

export async function translateText({
  text,
  source_lang = SOURCE_LANGUAGE.code,
  target_lang,
}: TranslateRequest): Promise<TranslateResponse> {
  const response = await fetch(`${TRANSLATION_API_BASE}/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source_lang, target_lang }),
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      typeof data.error === "string" ? data.error : "Translation request failed."
    );
  }

  return data as TranslateResponse;
}
