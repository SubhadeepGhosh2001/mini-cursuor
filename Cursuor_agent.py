import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()


client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

access_key = os.getenv("ACCESS_KEY")


def run_command(command):
    result = os.system(command=command)
    return result


def get_images(image: str):
    access_key = os.getenv("pixabay_key")  # Make sure this is set in your .env
    url = f"https://pixabay.com/api/?key={access_key}&q={image}&image_type=photo"
 
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        image_urls = [img["urls"]["regular"] for img in data["results"]]
        return image_urls
    else:
        return f"Error: {response.status_code} - {response.text}"


available_tools = {
    "run_command": {
        "fn": run_command,
        "description": "Takes a command as input to execute on system and returns output",
    },
    "get_images": {
        "fn": get_images,
        "description": "Accepts a search query as a string and retrieves a list of image URLs related to the query from the Unsplash API.",
    },
}


system_prompt = """
    You are a coding and a system assistant who is specialized in coding and debugging. You are also specialized in windows & linux commands and execute them into the system.
    You work on the basis of the user's commands and follow the user's instructions.
    You work on start, plan, action, observe mode.
    For the given query and available tools. And based on the tool selection you perform an action to call the tools.
    Wait for the observation and based on the observation from the tool call resolve the user query. 
    
    Rules:
    - Follow the output JSON format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query.
    - Don't create the similar files and folders again if it was already created.
    - If you are creating more than 2 file and folder execute another terminal for the command execution.
    - Also make sure you are using the correct code don't make mistakes in code.
    - Test the app or file once before ending the command and inform the user if everything is working fine.
    
    Output JSON Format:
    {{
        "step":"string",
        "content":"string",
        "function":"The name of function if the step is action",
        "input":"The input parameter for the function"
    }}
    
    Available Tools:
    - run_command: Takes a command as input to execute as input to execute on system and returns output
    - get_images: Takes the image name as string and fetches the images url from the web.
    
    Example:
    User Query: Create a folder name "New_Folder" in the current directory.
    output: {{"step":"plan","content":"The user wants to create a folder in the current directory as New_Folder"}}
    output: {{"step":"plan","content":"From the available tools I should call mkdir and write the user's requested folder name"}}
    output: {{"step":"action","function":"create_folder","Input":"Folder Name"}}
    output: {{"step":"observe","output":"New_Folder"}}
    output: {{"step":"output","content":"The folder has been successfully created."}}
    
    User Query: Make a E-commerce website in React-vite.
    output: {{"step":"plan","content":"The user wants to create a react-vite application"}}
    output: {{"step":"plan","content":"From the available tools I should the required commands to make the application"}}
    output: {{"step":"action","function":"run_command","Input":"Creates the application according to the user's requirement."}}
    output: {{"step":"observe","output":"The completely functional react-vite application"}}
    output: {{"step":"output","content":"The application has been created successfully."}}
    
    This will be the structure of the application:
    
    📦 src
├── 📁 assets             # Images, fonts, icons
├── 📁 components         # Reusable UI components
│   └── 📁 common         # Generic shared components (Button, Modal, Input)
├── 📁 constants          # App-wide constants (e.g., routes, configs)
├── 📁 features           # Feature-based structure (grouped by functionality)
│   ├── 📁 auth
│   │   ├── Login.jsx
│   │   ├── Signup.jsx
│   │   └── authSlice.js
│   ├── 📁 dashboard
│   │   ├── DashboardPage.jsx
│   │   └── dashboardSlice.js
│   └── ...
├── 📁 hooks              # Custom hooks (e.g., useAuth, useTheme)
├── 📁 layout             # Layout wrappers (Sidebar, Navbar, Footer)
├── 📁 lib                # Libraries/helpers (e.g., axios clients, validators)
├── 📁 pages              # Route-level pages (mapped to router)
│   ├── Home.jsx
│   ├── About.jsx
│   └── ...
├── 📁 router             # React Router configuration (protected routes, etc.)
├── 📁 services           # API calls (e.g., authService.js, userService.js)
├── 📁 store              # Global state management (Redux/Zustand/Context)
│   ├── index.js
│   └── slices/
├── 📁 styles             # Global styles, Tailwind or CSS files
│   └── global.css
├── 📁 utils              # Utility functions (e.g., debounce, formatDate)
├── App.jsx
├── main.jsx
└── vite-env.d.js (optional)

.vite.config.js
.env
tailwind.config.js

    
"""
messages = [{"role": "system", "content": system_prompt}]


while True:
    user_query = input("Enter the command: ")
    messages.append({"role": "user", "content": user_query})

    while True:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            response_format={"type": "json_object"},
            messages=messages,
        )

        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({"role": "assistant", "content": json.dumps(parsed_output)})

        if parsed_output.get("step") == "plan":
            print(f'🧠: {parsed_output.get("content")}')
            continue

        if parsed_output.get("step") == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")

            if available_tools.get(tool_name, False) != False:
                output = available_tools[tool_name].get("fn")(tool_input)
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps({"step": "observe", "output": output}),
                    }
                )
                continue

        if parsed_output.get("step") == "output":
            print(f'🤖: {parsed_output.get("content")}')
            break