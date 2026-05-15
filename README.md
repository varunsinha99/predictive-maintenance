# 🚀 Smart Predictive Maintenance System (SPMS)
**Project ID:** BTP2CSE145 | **Specialization:** Cloud Computing

### 🛠 The Problem
In manufacturing, equipment failure leads to massive downtime and financial loss. Most companies follow "Reactive Maintenance" (fix it when it breaks). This project shifts the paradigm to **"Predictive Maintenance"**—using Machine Learning to predict failures before they happen.

---

## 🏗 Project Architecture
This isn't just a Python script. It's a production-ready application containerized for the cloud.



- **Data Processing:** Scaled and cleaned sensor data (Air temperature, Process temperature, Rotational speed, Torque, Tool wear).
- **Machine Learning:** Trained a predictive model to classify failure risks.
- **Frontend:** Interactive dashboard built with **Streamlit**.
- **DevOps:** Fully containerized using **Docker** for "Build Once, Run Anywhere" capability.

---

## ⚡ Tech Stack
- **Languages:** Python 3.9
- **Libraries:** Scikit-learn, Pandas, NumPy, Streamlit
- **DevOps/Cloud:** Docker, Docker Hub, AWS EC2
- **Tools:** Git, VS Code

---

## 🐳 Dockerization (Cloud Ready)
This project is officially containerized. You don't need to install Python or any libraries locally—just run it via Docker.

**Public Image:** `varunsinha2112/spms-app`

### Quick Start with Docker:
1. **Pull the image from Docker Hub:**
   ```bash
   docker pull varunsinha2112/spms-app
