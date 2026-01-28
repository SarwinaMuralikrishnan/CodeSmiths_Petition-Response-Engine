from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load trained AI model and vectorizer
model = pickle.load(open("petition_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

# User page
@app.route("/")
def user_page():
    return render_template("user.html")

# AI prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    description = data["description"]

    # Convert text to numbers
    text_vec = vectorizer.transform([description])

    # Predict department
    department = model.predict(text_vec)[0]

    return jsonify({"department": department})

# Admin page
@app.route("/admin")
def admin_page():
    return render_template("admin.html")

# Run app
if __name__ == "__main__":
    app.run(debug=True)
