from openai import OpenAI
import os
import json
import requests
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai


load_dotenv()

# client = wrappers.wrap_openai(OpenAI(
#     api_key=os.getenv("GROQ_API_KEY"),
#     base_url="https://api.groq.com/openai/v1"
# ))

client = wrap_openai(OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
))
@traceable(name="Get Weather")
def get_weather(city:str):
    print("Tool Called: get_weather",city)

    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code==200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong!"

@traceable(name="Run Command")
def run_command(command):
    result = os.system(command=command)
    return result

@traceable(name="Get Average")
def get_average(*temps):
    print("Tool Called: get_average",temps)
    total = 0
    for i in temps:
        total += int(i)
    
    return total / len(temps)

available_tools={
    "get_weather":{
        "fn":get_weather,
        "description": "Takes a city name as an input and returns the current weather for the city"
    },
    "run_command":{
        "fn":run_command,
        "description": "Take command as input and execute to system and return output"
    }
}


system_prompt = f"""
    Yout are an helpfull AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe node.
    For the given user query and available tools, plan the step bu stem execution, based on the planning,
    select the relevent tool from the available tool, and based on the tool selection you perform an action to call the tool.
    wait for the observation and based on the observation from the tool call resolve the user quer.

    Rules:
    - If weather information is requested you MUST call get_weather.
    - Never generate weather data yourself.
    - Weather information can ONLY come from the get_weather tool.
    - If User provide command then carefully execute it on the terminal use the run_command tool.

    Output JSON Format:
    {{
        "step":"string",
        "content":"string",
        "function":"The name of function if the step is action",
        "input":"The input parameter for the function"
    }}

    Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city
    - run_command: Take command as input and execute to system and return output 
    
    Example 1:
    User Query: What is the weather of new york?
    Output:{{"step":"plan", "content":"The user is interseted in weather data of new york"}}
    Output:{{"step":"plan", "content":"From the available tools I should call get_weather function"}}
    Output:{{"step":"action","function":"get_weather","input":"new york"}}
    Output:{{"step":"observe","output":"12 Degree Cel"}}
    Output:{{"step":"output", "content":"The weather of new york seems to be 12 degrees."}}

    Example 2:
    Output:{{"step":"plan","content":"The user wants to push the current project to GitHub and create a new repository named weather-agent"}}
    Output:{{"step":"plan","content":"To do this I need to initialize git, add project files, commit them and create the repository on GitHub"}}
    Output:{{"step":"action","function":"run_terminal","input":"git init"}}
    Output:{{"step":"observe","output":"Initialized empty Git repository"}}
    Output:{{"step":"action","function":"run_terminal","input":"git add ."}}
    Output:{{"step":"observe","output":"All project files staged"}}
    Output:{{"step":"action","function":"run_terminal","input":"git commit -m 'initial commit'"}}
    Output:{{"step":"observe","output":"Files committed successfully"}}
    Output:{{"step":"action","function":"run_terminal","input":"gh repo create weather-agent --public --source=. --push"}}
    Output:{{"step":"observe","output":"Repository weather-agent created and code pushed to GitHub"}}
    Output:{{"step":"output","content":"Your project has been successfully pushed to a new GitHub repository named weather-agent"}}
"""

message=[
    {"role":"system","content":system_prompt}
]

while True:
    user_query=input('> ')
    if user_query=="exit":
        print("Ok Bye, See you Again!")
        break

    message.append({"role":"user","content":user_query})

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            # response_format={"type":"json_object"},
            messages=message
        )

        parsed_output = json.loads(response.choices[0].message.content)
        message.append({"role":"assistant","content":json.dumps(parsed_output)})

        if isinstance(parsed_output, list):
            parsed_output = parsed_output[0]
            print(f"Brain: {parsed_output.get("content")}")
            break
        elif parsed_output.get("step") == "plan":
            print(f"Brain: {parsed_output.get("content")}")
            continue

        if parsed_output.get("step") == "action":
            tool_name = parsed_output.get("function")
            tool_input= parsed_output.get("input")

            if available_tools.get(tool_name, False) !=False:
                output= available_tools[tool_name].get("fn")(tool_input)
                message.append({"role":"assistant", "content":json.dumps({"step":"observe", "output":output})})
                continue
        
        if parsed_output.get("step") == "output":
            print(f"Yeh!!:{parsed_output.get("content")}")
            break

