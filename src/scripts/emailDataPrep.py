import random
import os
import pandas as pd

# -----------------------
# Phrase banks
# -----------------------
information_query_phrases = [
    "I would like to know about the {service} service availability in {location}.",
    "Can you provide details about the {service} offered by {business}?",
    "What are the operating hours of {business}?",
    "Could you tell me if {business} supports {service_name}?",
    "I need information about the {feature_name} feature in the app.",
    "Can you give me more information about {hotel_name} in {city}?"
]

actionable_phrases = [
    "Please update my business listing for {business_name}.",
    "Kindly change the operating hours to {new_hours}.",
    "Please correct the category to {correct_category}.",
    "Add {new_service} to my services.",
    "Update my address to {new_address}."
]

existing_issue_phrases = [
    "I am facing issues logging into my account.",
    "The payment process is failing repeatedly.",
    "I cannot update my profile details on the portal.",
    "My subscription is not reflecting correctly.",
    "There seems to be a bug in the latest app update."
]

# Short email phrase banks
short_information_query = [
    "Need business hours of {business}.",
    "Phone number of {business}?",
    "Does {business} deliver?",
    "Looking for a {service} in {location}.",
    "Rating for {hotel_name}?"
]

short_actionable = [
    "Update my phone number.",
    "Change business hours to {new_hours}.",
    "Add service {new_service}.",
    "Correct category to {correct_category}.",
    "Change address to {new_address}."
]

short_existing_issue = [
    "Login not working.",
    "Payment failed.",
    "Account suspended.",
    "Invoice not generating.",
    "Verification code not received."
]

# -----------------------
# Helpers
# -----------------------
def generate_service():
    return random.choice(["plumbing", "catering", "web hosting", "delivery", "consulting"])

def generate_location():
    return random.choice(["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad"])

def generate_business_name():
    return random.choice(["ABC Corp", "Sunshine Services", "Tech Solutions", "Global Mart", "Fresh Foods"])

def add_typos(text):
    typo_samples = {"service":"servcie", "address":"adress", "business":"busines", "account":"acount"}
    for k, v in typo_samples.items():
        if k in text and random.random() < 0.3:
            text = text.replace(k, v)
    return text

# -----------------------
# Email generator
# -----------------------
def generate_email(category):
    is_short = random.random() < 0.5  # 50% short emails

    if category == "information_query":
        template = random.choice(short_information_query) if is_short else random.choice(information_query_phrases)
        body = template.format(
            service=generate_service(),
            location=generate_location(),
            business=generate_business_name(),
            service_name="JD Pay",
            feature_name="Book a table",
            business_name=generate_business_name(),
            hotel_name="The Grand Hotel",
            city="Goa",
            new_hours="9 AM - 7 PM",
            correct_category="IT Services",
            new_service="web design",
            new_address="123 New Road, New Delhi"
        )

    elif category == "actionable":
        template = random.choice(short_actionable) if is_short else random.choice(actionable_phrases)
        body = template.format(
            business_name=generate_business_name(),
            new_hours="9 AM - 7 PM",
            correct_category="IT Services",
            new_service="web design",
            new_address="123 New Road, New Delhi"
        )

    elif category == "existing_issue":
        template = random.choice(short_existing_issue) if is_short else random.choice(existing_issue_phrases)
        body = template.format()

    # Add email signature for long ones
    if not is_short:
        body = f"Hi Team,<br>{body}<br><br>Thanks,<br>{random.choice(['Sunil','Kavya','Rohit','Anjali','Gautam'])}"

    # Add typos sometimes
    if random.random() < 0.2:
        body = add_typos(body)

    subject = f"Subject: {random.choice(['Query','Help needed','Request','Issue'])}"
    return f"{subject}<br>{body}", category


# -----------------------
# Dataset generator
# -----------------------
def generate_dataset(n=300, out_path="data/emails.csv"):
    categories = ["information_query", "actionable", "existing_issue"]
    data = []

    for _ in range(n):
        category = random.choice(categories)
        email, label = generate_email(category)
        data.append({"email": email, "label": label})

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Generated dataset with {len(df)} emails at {out_path}")


if __name__ == "__main__":
    generate_dataset(300)
