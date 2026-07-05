export const SOURCE_LANGUAGE = {
  name: "English",
  code: "eng_Latn",
} as const;

export const LANGUAGES = [
  { name: "Assamese", code: "asm_Beng" },
  { name: "Bengali", code: "ben_Beng" },
  { name: "Bodo", code: "brx_Deva" },
  { name: "Dogri", code: "doi_Deva" },
  { name: "Gujarati", code: "guj_Gujr" },
  { name: "Hindi", code: "hin_Deva" },
  { name: "Kannada", code: "kan_Knda" },
  { name: "Kashmiri (Arabic)", code: "kas_Arab" },
  { name: "Kashmiri (Devanagari)", code: "kas_Deva" },
  { name: "Konkani", code: "gom_Deva" },
  { name: "Maithili", code: "mai_Deva" },
  { name: "Malayalam", code: "mal_Mlym" },
  { name: "Marathi", code: "mar_Deva" },
  { name: "Manipuri (Bengali)", code: "mni_Beng" },
  { name: "Manipuri (Meitei)", code: "mni_Mtei" },
  { name: "Nepali", code: "npi_Deva" },
  { name: "Odia", code: "ory_Orya" },
  { name: "Punjabi", code: "pan_Guru" },
  { name: "Sanskrit", code: "san_Deva" },
  { name: "Santali", code: "sat_Olck" },
  { name: "Sindhi (Arabic)", code: "snd_Arab" },
  { name: "Sindhi (Devanagari)", code: "snd_Deva" },
  { name: "Tamil", code: "tam_Taml" },
  { name: "Telugu", code: "tel_Telu" },
  { name: "Urdu", code: "urd_Arab" },
] as const;

export type TargetLanguageCode = (typeof LANGUAGES)[number]["code"];
