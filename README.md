# Medical Specialty Predictor (NLP Healthcare ML App)

This project predicts the **medical specialty associated with a clinical report** using Natural Language Processing and Machine Learning.

The system analyzes clinical transcription text and classifies it into specialties such as:

* Cardiology
* Orthopedic
* Radiology
* Neurology
* Gastroenterology
* Surgery
* and more.

The model also provides **Explainable AI insights** by highlighting the most influential keywords that contributed to the prediction.

---

# Live Demo

Try the deployed application here:

https://medical-specialty-ai.streamlit.app

---

# Tech Stack

* Python
* Scikit-learn
* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Naive Bayes classifier
* Streamlit
* Matplotlib

---

# How It Works

1. Clinical transcription text is preprocessed using NLP techniques
2. Text is converted into numerical features using **TF-IDF vectorization**
3. A **Logistic Regression classifier** predicts the most likely medical specialty
4. The system displays prediction probabilities and key words influencing the prediction

---

# Sample Inputs for Testing

You can copy and paste the following clinical notes into the app.

### Cardiology Example

The patient is a 58-year-old female presenting with chest pain and shortness of breath on exertion. Past medical history includes hypertension and hyperlipidemia. EKG revealed ST-T wave abnormalities. Echocardiogram showed mild left ventricular hypertrophy with preserved ejection fraction. The patient was admitted for cardiac evaluation and stress testing to rule out ischemic heart disease.

### Orthopedic Example

The patient is a 67-year-old male with severe right knee pain due to advanced osteoarthritis. Conservative treatment including physical therapy and anti-inflammatory medications failed. The patient underwent right total knee arthroplasty. Degenerative cartilage was removed and prosthetic femoral and tibial components were placed during surgery.

### Neurology Example

A 45-year-old female presents with recurrent headaches and dizziness. Neurological examination revealed decreased sensation in the left arm but normal motor strength. MRI of the brain showed no acute intracranial abnormality. Neurology consultation recommended migraine prophylaxis and outpatient follow-up.

### Radiology Example

CT scan of the abdomen and pelvis with contrast shows normal liver, spleen, pancreas, and kidneys. No evidence of bowel obstruction or intra-abdominal mass is identified. Mild diverticulosis is noted without evidence of diverticulitis.

### Gastroenterology Example

The patient is a 52-year-old male with abdominal pain, nausea, and intermittent rectal bleeding. Colonoscopy revealed inflammatory changes in the descending colon. Biopsy samples were obtained for further evaluation and the patient was started on proton pump inhibitor therapy.
