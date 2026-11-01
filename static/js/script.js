document.addEventListener("DOMContentLoaded", function () {

    console.log("Student Performance Prediction System Loaded");

    const form = document.querySelector(".prediction-form");

    if (form) {

        form.addEventListener("submit", function () {

            const button = form.querySelector("button");

            button.textContent = "Generating Prediction...";

            button.disabled = true;

        });

    }

});