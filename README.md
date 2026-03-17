# 📊 JAVAX

A backend system that generates charts from dataset based on user prompts.

---

## 🚀 Setup Instructions

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd <your-project-folder>
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup MongoDB

#### Option 1: MongoDB Atlas (Recommended)

- Create account on MongoDB Atlas  
- Create a cluster  
- Create DB user  
- Allow network access (0.0.0.0/0)  
- Copy connection string  

Example:
```bash
mongodb+srv://<username>:<password>@cluster0.mongodb.net/<dbname>
```

#### Option 2: Local MongoDB

- Install MongoDB Community Server  

Start server:
```bash
mongod
```

Default URI:
```bash
mongodb://localhost:27017
```

### 5. Setup Environment Variables

Create `.env` file in root:

```env
OPENAI_API_KEY=your_api_key
MONGO_URI=your_mongodb_connection_string
```

### 6. Run Application
FastAPI:
```bash
python uvicorn api:app --reload
```

---

## 📂 Input

- Place dataset in project folder  
- Supported format: `.csv`  

---

## ⚙️ Usage

Example prompts:
```text
dashboard
sales trends
top categories
```


---

## 📊 Output

- Charts are generated and stored in the `/charts` folder  
- Each chart is saved as a PNG image  
- Chart file paths are stored in the database (MongoDB)  
- The backend reads these images and converts them into Base64 format  
- The API returns Base64-encoded images to the frontend

> Note: Images are returned as Base64 strings, not direct URLs on Frontend.

---

## 📁 Project Structure

```text
project/
│── api.py
│── generate_charts.py
│── charts/
│── .env
│── requirements.txt
```

---

## ❗ Common Issues

### MongoDB not connecting
- Check `MONGO_URI`  
- Verify username/password  
- Ensure network access enabled (Atlas)  

### Charts not generating
- Dataset issue  
- Prompt format issue  
