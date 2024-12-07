# **Function Calling Test Suite**

A customizable test suite for evaluating the functionality and tool-handling capabilities of the **Llama Series** models using the `ChatOllama` wrapper. This test suite allows you to test the model's performance on real-world scenarios, such as weather queries, and provides a detailed success rate analysis.

---

## **Features**

- **Dynamic Test Case Generation**: Generate test cases with customizable sizes.
- **Tool Integration**: Evaluate the model's ability to call external tools (e.g., weather API, system time).
- **Progress Tracking**: Monitor execution with a progress bar.
- **Comprehensive Output**: Logs detailed results for each test case.
- **Error Handling**: Gracefully handles exceptions without interrupting the test flow.
- **Result Persistence**: Saves test results in a structured JSON file for further analysis.

---

## **Installation**

### **Prerequisites**
1. Python 3.8 or above.
2. Install the required Python packages

3. **Tools Dependency**:
   - Ensure `tools.py` contains the necessary tool implementations, e.g., `get_current_weather` and `get_system_time`.

4. **Llama Model Server**:
   - Host the Llama 3.3 model on a local or remote server accessible via an API. Update the `base_url` in the script accordingly.

---

## **Usage**

### **Running the Test Suite**
1. Clone this repository.

2. Run the test suite:
   ```bash
   python function_calling_test_suite.py
   ```

3. When prompted, enter the desired number of test cases:
   ```
   Enter the number of test cases to generate for the test suite: 50
   ```

4. The test suite will execute and provide live updates, including:
   - Test sentence being evaluated.
   - AI response and tool calls.
   - Tool outputs.

5. At the end of execution, results will be saved in `test_results.json`.

---

## **Configuration**

### **Customizing Test Sentences**
You can modify the list of locations and query templates in the script to test different scenarios:

- **Locations** (`cities_and_states`):
  ```python
  cities_and_states = [
      "New York, NY", "Los Angeles, CA", ...
  ]
  ```

- **Queries** (`specific_requests`):
  ```python
  specific_requests = [
      "What is the weather today in [location]?",
      "Can you tell me the forecast for tomorrow in [location]?",
      ...
  ]
  ```

### **Adjusting the Model Configuration**
Modify the `base_url` and `model` fields to match your hosted Llama server:
```python
model = ChatOllama(
    base_url="http://your-server-address:port",
    model="llama3.3",
    format="json"
)
```

---

## **Output**

### **Terminal Output**
The terminal will display:
- Each test case and its corresponding response.
- Tool calls made by the model, along with their outputs.
- Errors encountered during the test suite execution.

### **Result File**
A `test_results.json` file is generated containing detailed results, including:
- The test sentence.
- AI response.
- Tool calls and outputs.
- Errors, if any.

---

## **Sample Output**

### **Terminal Sample**
```
Generated Test Sentences:
- What is the weather today in New York, NY?
- How's the weather in Los Angeles, CA today?
...

Running Test Suite...

Test #1: What is the weather today in New York, NY?
AI Response: {"text": "It is sunny and 75°F."}
Tool Invoked: get_current_weather with args {'location': 'New York, NY'}
Tool Output: {'temperature': '75°F', 'condition': 'sunny'}

Test #2: How's the weather in Los Angeles, CA today?
AI Response: {"text": "It is cloudy and 68°F."}
No tools invoked by the model.
...
Test Suite Completed.
Total Sentences Tested: 50
Function Calls Made: 25
Success Rate: 50.00%
```

### **Result File Sample (`test_results.json`)**
```json
[
  {
    "sentence": "What is the weather today in New York, NY?",
    "result": "It is sunny and 75°F.",
    "tool_calls": [
      {
        "name": "get_current_weather",
        "args": {"location": "New York, NY"}
      }
    ]
  },
  {
    "sentence": "How's the weather in Los Angeles, CA today?",
    "result": "It is cloudy and 68°F.",
    "tool_calls": []
  }
]
```

---

## **Customization Tips**

### **Adding New Tools**
1. Implement the tool in `tools.py`:
   ```python
   @tool
   def new_tool(args):
       # Tool logic here
       return output
   ```
2. Add the tool to the `bind_tools` list and `tool_mapping` dictionary.

### **Expanding Test Scenarios**
Update the `cities_and_states` or `specific_requests` lists to include more diverse inputs.

---

## **Contributing**
We welcome contributions! Feel free to:
- Add more tools or test cases.
- Report bugs or suggest enhancements by creating an issue.
