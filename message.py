import smtplib
from email.message import EmailMessage


def send_email(subject, body, to):
    user = "dogg.bott1@gmail.com"
    password = "lqhqbwbkpurtnjuj"

    msg = EmailMessage()
    msg.set_content(body)
    msg["subject"] = subject
    msg["to"] = to
    msg["from"] = user

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(user, password)
    server.send_message(msg)
    server.quit()
