from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import pickle
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
    conn.commit()
    conn.close()

init_db()
model = pickle.load(open("petition_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/user")
def user_page():
    if not session.get("user_logged_in"):
        return redirect(url_for("login_page", type="public"))
    return render_template("user.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    description = data["description"]

    text_vec = vectorizer.transform([description])

    department = model.predict(text_vec)[0]

    return jsonify({"department": department})

@app.route("/admin")
def admin_page():
    if not session.get("admin_logged_in"):
        return redirect(url_for("login_page", type="admin"))
    return render_template("admin.html")

@app.route("/api/login", methods=["POST"])
def auth_login():
    data = request.json
    login_type = data.get("type")
    userid = data.get("userid")
    password = data.get("password")

    if login_type == "admin":
        if userid == "admin@tn.gov.in" and password == "Admin@2026":
            session["admin_logged_in"] = True
            return jsonify({"success": True, "redirect": "/admin"})
        else:
            return jsonify({"success": False, "message": "Invalid Admin Credentials"}), 401
    else:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (userid, password))
        user = c.fetchone()
        conn.close()
        
        if user:
            session["user_logged_in"] = True
            return jsonify({"success": True, "redirect": "/user"})
        else:
            return jsonify({"success": False, "message": "Invalid Credentials. Please sign up or check your details."}), 401

@app.route("/api/signup", methods=["POST"])
def auth_signup():
    data = request.json
    userid = data.get("userid")
    password = data.get("password")

    if not userid or not password or len(userid) < 5 or len(password) < 4:
         return jsonify({"success": False, "message": "ID must be 5+ characters and PIN 4+ characters"}), 400

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, 'public')", (userid, password))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Registration Successful! Please login."})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"success": False, "message": "User ID already exists!"}), 400

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index_page"))

@app.route("/api/notify", methods=["POST"])
def notify_citizen():
    data = request.json
    email = data.get("email")
    petition_id = data.get("id")
    status = data.get("status")

    if not email:
        return jsonify({"success": False, "message": "No email provided."})

    remarks = data.get("remarks", "No official remarks provided.")
    days_taken = data.get("daysTaken", "N/A")

    print("\n" + "="*50)
    print(f"📧 NEW SIMULATED EMAIL NOTIFICATION")
    print(f"TO: {email}")
    print(f"SUBJECT: Update on your Grievance Petition: {petition_id}")
    print("-"*50)
    
    if status == "Resolved":
        print(f"Dear Citizen,\n")
        print(f"We are pleased to inform you that your petition ({petition_id}) has been successfully RESOLVED.")
        print(f"Resolution Time: {days_taken} day(s)")
        print(f"Official Remarks / Action Taken:\n> {remarks}")
    elif status == "In Progress":
        print(f"Dear Citizen,\n")
        print(f"Your petition ({petition_id}) is currently IN PROGRESS and is being actively reviewed by officials.")
        print(f"Official Remarks:\n> {remarks}")
    else:
        print(f"Dear Citizen,\n")
        print(f"Your petition ({petition_id}) has been updated to: {status}.")
        print(f"Official Remarks:\n> {remarks}")
        
    print("="*50 + "\n")
    
    return jsonify({"success": True, "message": f"Email successfully dispatched to {email}"})

if __name__ == "__main__":
    app.run(debug=True)