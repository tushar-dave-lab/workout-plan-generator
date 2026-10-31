import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import streamlit as st
from fastapi import FastAPI, HTTPException, Query
from groq import Groq


load_dotenv()

# Set up page configuration
st.set_page_config(page_title="Workout Plan Generator", page_icon="🏋️‍♂️", layout="centered")

def generate_workout_plan(
    goal: str,
    experience: str,
    days: int,
    equipment: list[str],
    limitations: Optional[str] = None
) -> str:
    """
    Constructs a detailed prompt, makes an API call to Groq, 
    and handles potential runtime or API execution issues.
    """
   
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY in environment")

    # Initialize Groq client using environment variable or streamlit secrets
    # It automatically looks for os.environ.get("GROQ_API_KEY")
    client = Groq()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    # Create the structured system prompt to enforce rules
    system_prompt = (
        "You are an expert, elite personal trainer and strength coach.\n"
        "Your job is to generate highly personalized, realistic, and actionable weekly workout plans.\n\n"
        "CRITICAL RULES:\n"
        "1. Strictly respect all provided constraints (days available, equipment access, and limitations).\n"
        "2. Never recommend exercises requiring equipment the user does not have access to.\n"
        "3. Output MUST be neatly formatted with Day-by-Day headers, clear exercises, sets, and repetitions.\n"
        "4. DO NOT make any definitive medical claims. If physical limitations/injuries are mentioned, "
        "you must include a prominent, brief safety disclaimer at the top of your response advice."
    )

    # Format user inputs into a structured payload
    equipment_str = ", ".join(equipment) if equipment else "No equipment"
    limitations_str = limitations if limitations and limitations.strip() else "None reported"

    user_prompt = f"""
    Please generate a custom weekly workout plan based on these constraints:
    - **Fitness Goal**: {goal}
    - **Experience Level**: {experience}
    - **Days Available Per Week**: {days} days
    - **Equipment Access**: {equipment_str}
    - **Physical Limitations/Injuries**: {limitations_str}
    
    Provide a day-by-day structural breakdown with specific exercises, target sets, and rep ranges.
    """

    try:
        # Execute the Groq API call (using llama-3.3-70b-versatile or a current fast model)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=2048
        )
        
        response_text = completion.choices[0].message.content
        
        if not response_text:
            raise ValueError("The AI returned an empty response string.")
            
        return response_text

    except Exception as e:
        # Gracefully handle API down, bad key, network dropouts, or malformed data
        raise RuntimeError(f"Failed to generate plan securely. Details: {str(e)}")

# --- Streamlit UI App Layout ---
st.title("🏋️‍♂️ Personalized Workout Plan Generator")
st.write("Provide your exact training parameters to generate a custom fitness routine.")

# 1. Structured UI Inputs
goal = st.selectbox(
    "What is your primary fitness goal?",
    ["Build muscle", "Lose fat", "General fitness", "Improve endurance"]
)

experience = st.selectbox(
    "What is your current experience level?",
    ["Beginner", "Intermediate", "Advanced"]
)

days = st.slider("How many days per week can you train?", min_value=1, max_value=7, value=3)

equipment = st.multiselect(
    "What equipment do you have access to?",
    ["No equipment", "Home dumbbells", "Full gym"],
    default=["No equipment"]
)

limitations = st.text_area(
    "Any injuries, physical limitations, or exercises to avoid? (Optional)",
    placeholder="e.g., Bad lower back, no overhead shoulder pressing, recovering from a sprained wrist"
)

# 2. State & Button Trigger Validation logic
if st.button("Generate Plan", type="primary"):
    # Basic input error protection checks
    if days == 0:
        st.error("Please select at least 1 day per week to generate a plan.")
    elif not equipment:
        st.error("Please pick at least one equipment choice (even if it's 'No equipment').")
    else:
        with st.spinner("Analyzing parameters and engineering your plan..."):
            try:
                # Execution
                workout_plan = generate_workout_plan(
                    goal=goal,
                    experience=experience,
                    days=days,
                    equipment=equipment,
                    limitations=limitations
                )
                
                # Success display
                st.success("Plan generated successfully!")
                st.markdown("---")
                st.markdown(workout_plan)
                
            except Exception as error_msg:
                # Assignment fallback requirement: user friendly errors
                st.error("⚠️ Connection Error or Server Issue")
                st.info(
                    "We couldn't connect to the backend AI server. Please make sure your "
                    "api key is properly added to your environment variables or Streamlit secrets."
                )
