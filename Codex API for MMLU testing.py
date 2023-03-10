# -----------------------------------------------------------
from time import sleep
import os
import csv
import openai
# Set your OpenAI API key
openai.api_key = "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"
#"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"

# Set the path to the folder containing the input files
folder_path = "C:/Users/aksha/Desktop/MMLU-Data/test"

# C:\Users\\\MMLU-Data\test

# Iterate through the files in the folder
for filename in os.listdir(folder_path):

    if filename.endswith(".csv"):

        # Set the path to the input file
        input_file_path = os.path.join(folder_path, filename)

        # Set the path to the output file
        output_file_path = os.path.join(
            folder_path, f"{os.path.splitext(filename)[0]}_results.csv"
        )

        # Open the input file and create a CSV reader object
        with open(input_file_path, "r") as input_file:
            reader = csv.reader(input_file)

            # Open the output file and create a CSV writer object
            with open(output_file_path, "w", newline="") as output_file:
                writer = csv.writer(output_file)

                # Iterate through the rows in the input file
                for row in reader:
                    sleep(0.5)
                    
                    # Convert the row data to a string
                    data = ",".join(row)

                    # Send the data to the OpenAI Codex API
                    response = openai.Completion.create(
                        engine="davinci-codex",
                        prompt=data,
                        max_tokens=256,
                        n=1,
                        stop=None,
                        temperature=0,
                    )

                    # Extract the generated text from the response
                    generated_text = response.choices[0].text.strip()

                    # Write the generated text to the output file
                    writer.writerow([generated_text])
                    sleep(0.5)
