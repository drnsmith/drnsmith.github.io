---
date: 2025-03-20T10:58:08-04:00
publishDate: 2024-03-20T10:58:08-04:00
description: "AI-Powered Recipe Assistant: Generate, Customise, and Save Recipes Using AI."
image: "/images/project14_images/project14.png"
tags: ["AI", "Machine Learning", "Python", "Streamlit", "LLMs"]
title: "AI-Powered Recipe Assistant: Smart Cooking with AI"
---

{{< figure src="/images/project14_images/project14.png">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/AI-Assistant-Recipe" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction  

The **AI-Powered Recipe Assistant** is an interactive tool designed to help users **generate, customise, and save recipes** using AI. It leverages **large language models (LLMs)** to dynamically create recipes based on user-provided ingredients and preferences.  

Built with **Streamlit**, the assistant offers a clean UI where users can:  
 - Find recipes from a curated database.  
 - Generate AI-created recipes when no match is found.  
 - Save and manage favourite recipes.  

This project demonstrates the **fusion of AI, NLP, and data-driven decision-making**, transforming how users interact with recipe recommendations.  

### Key Features  

#### 1. Ingredient-Based Recipe Search 
- Retrieve **real-world recipes** from a database using simple keyword searches.  

#### 2. AI-Generated Recipes 
- If no recipe is found, the assistant **creates a new one** using **Ollama's Llama3 model**.  
- Ensures diversity and flexibility, catering to various dietary preferences.  

#### 3. Save & Manage Recipes* 
- Users can **save their favourite recipes** and retrieve them later.  

#### 4. Dynamic UI & Real-Time Feedback  
- Built with **Streamlit**, offering an intuitive experience with real-time updates.  


### Tech Stack  

- **Frontend:** Streamlit (for the user interface)  
- **Backend:** Python (FastAPI in future iterations)  
- **LLMs:** Ollama (Llama3 model)  
- **Database:** SQLite (for recipe storage)  
- **Deployment:** GitHub Pages (planned), Docker (future)  

### How It Works  

#### 1. Searching for a Recipe  

Users enter ingredients, and the system:  
- **Looks up a relevant recipe** in the database.  
- **Displays the recipe** if a match is found.  
- If no match, it **suggests generating an AI recipe** instead.  

#### 2. AI Recipe Generation  

If no recipe is found, an AI-generated recipe is created using **LLMs**:  

```python
def generate_ai_recipe(ingredients):
    prompt = f"Create a detailed recipe using these ingredients: {ingredients}. Format it as follows:\n\n"
    ai_response = llm.invoke(prompt)
    return parse_ai_response(ai_response)
```
	•	The AI structures the recipe with ingredients, instructions, and preparation steps.
	•	Recipes are designed to be human-readable and easy to follow.


#### 3. Saving Recipes

Users can save AI-generated or database recipes to their personal collection:
```python
def save_recipe(title, ingredients, instructions):
    cursor.execute("INSERT INTO saved_recipes (title, ingredients, instructions) VALUES (?, ?, ?)", 
                   (title, ingredients, instructions))
```
	•	Stored recipes appear under "Saved Recipes".
	•	Users can delete unwanted recipes anytime.


#### Challenges & Solutions

- Ensuring Recipe Diversity
**Problem**: AI-generated recipes sometimes repeated similar structures.
**Solution**: Added randomisation and diverse examples to prompts.

• Efficient Database Search
**Problem**: Searching millions of recipes for matching ingredients required optimisation.
**Solution**: Used SQL indexing and caching for faster lookups.

• Streamlit Button Issues
**Problem**: The “Find Another Recipe” button initially failed to refresh results.
**Solution**: Implemented session state tracking to trigger updates.

#### Future Enhancements

- **Improve AI Recipe Understanding**:
  
	 • Use embedding models for better ingredient matching.

	 • Improve recipe formatting for clarity.

- **API Integration**:
  
	 • Convert into a FastAPI service for recipe generation.

	 • Add external APIs for ingredient substitution and nutrition info.

#### Conclusion

The AI-Powered Recipe Assistant showcases the power of AI-driven decision-making in daily life. Whether retrieving a classic dish or generating a custom recipe, this tool provides a seamless user experience, blending data, NLP, and automation.

*Check out the project on GitHub and connect with me on LinkedIn to discuss AI in food tech!*
