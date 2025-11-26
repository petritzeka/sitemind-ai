# SiteMind AI

SiteMind AI is a WhatsApp-based AI assistant designed for UK electricians and electrical installation students. It delivers instant support for EAL Level 2/3 learning, domestic installation guidance, distribution board analysis, and automated documentation — all directly through WhatsApp.

Built using Flask, Twilio WhatsApp API, OpenAI GPT models, and a custom RAG system, SiteMind AI aims to reduce admin time, improve learning outcomes, and modernise the workflow of students and working electricians.

---

## Features

### Tutor Mode (EAL Level 2 & 3)
- Explanations aligned with EAL syllabus  
- BS 7671-style reasoning  
- Quick quizzes and revision plans  
- Testing procedures and step-by-step guides  

### Distribution Board OCR
- Send a photo of a consumer unit  
- AI extracts circuit numbers, breaker types, ratings, cable sizes, descriptions  
- Outputs a clean, structured circuit details table  

### Test-Sheet PDF Generator
- Generates Schedule of Circuit Details (Page 1)  
- Generates Schedule of Test Results (Page 2)  
- Uses extracted OCR data or user-provided values  
- PDFs match standard UK electrical documentation format  

### Voice-Note Analysis
- Users can send voice-notes describing faults or questions  
- Automatic transcription + technical explanation  

### Quotes & Invoices
- Instant quote and invoice generator  
- Ready-to-send client messages for electricians  

### Stripe Subscription Access Control
- Monthly/annual plans  
- Free trial logic  
- Webhooks to verify active subscribers  
- Blocks premium features for unpaid users  

### Multi-language Support
- English, Albanian, and Ukrainian auto-detection  
- Replies in the user's language  

---

## Tech Stack

- **Python (Flask)** — Backend  
- **Twilio WhatsApp API** — Messaging interface  
- **OpenAI GPT models** — AI responses  
- **Custom RAG System** — EAL + BS 7671 knowledge  
- **SQLite (PostgreSQL-ready)** — Local storage  
- **ReportLab** — PDF generation  
- **Google Vision / OCR pipeline** — Image analysis  
- **Stripe** — Subscription management  

---

## Project Structure

sitemind-ai/
│
├── app/
│ ├── routes/ # WhatsApp, Stripe, OCR, PDF endpoints
│ ├── services/ # AI logic, vectorstore, OCR pipeline
│ ├── pdf_templates/ # PDF creation & coordinates JSON
│ └── ...
│
├── rag/ # RAG documents for Tutor Mode & BS 7671
│
├── db/ # Database folder (SQLite, ignored in Git)
│
├── run.py # Entry point
├── requirements.txt # Python dependencies
└── .gitignore


---

## Installation & Setup

### 1. Clone the repo

git clone https://github.com/petritzeka/sitemind-ai.git

cd sitemind-ai


### 2. Create a virtual environment

python3 -m venv venv
source venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt


### 4. Add environment variables  
Create a `.env` file:

OPENAI_API_KEY=your_key
TWILIO_AUTH_TOKEN=your_token
TWILIO_ACCOUNT_SID=your_sid
WHATSAPP_NUMBER=your_twilio_whatsapp_number
STRIPE_SECRET_KEY=your_stripe_key


### 5. Run the server

python run.py
---

## Roadmap

- Improve OCR accuracy for poor-quality board photos  
- Add full BS 7671 chapter referencing  
- Add AM2 practice mode  
- Add Level 3 design calculations engine  
- Add analytics dashboard for college tutors  
- Deploy on Render/Railway with PostgreSQL  

---

## License

This project is licensed under the MIT License.

---

## Contact

Built by **Petrit Zeka**  

