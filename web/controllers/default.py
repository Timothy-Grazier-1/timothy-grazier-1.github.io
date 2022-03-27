from web import app

from flask import redirect, render_template


# Home Page Display
@app.route('/')
def index():
    return render_template('home.html')


# Redirects anyone who clicks on about to my personal site
@app.route('/bookme')
def about():
    return redirect('https://docs.google.com/forms/d/e/1FAIpQLSev2c4QjQj3a8yfGfNTd0bgW6p2w5_o5fqmkvyk3Cr7JEI-SA/viewform?usp=sf_link')
