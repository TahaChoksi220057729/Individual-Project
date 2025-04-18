import os
import PyPDF2
from collections import defaultdict
import re

def extract_pdf_text(directory_path):
    """
    Extract text from all PDF files in a directory and organize them.
    
    Args:
        directory_path (str): Path to directory containing PDF files
    
    Returns:
        dict: Dictionary with PDF names as keys and extracted text as values
    """
    # Dictionary to store results
    pdf_contents = defaultdict(str)
    
    # Regular expression to extract numbers from filenames
    number_pattern = re.compile(r'\d+')
    
    # Find all PDF files in directory
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    # Process each PDF file
    for pdf_file in pdf_files:
        file_path = os.path.join(directory_path, pdf_file)
        
        try:
            # Open PDF file
            with open(file_path, 'rb') as file:
                # Create PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from each page
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"
                
                # Store extracted text
                pdf_contents[pdf_file] = text.strip()
                
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    
    return pdf_contents

def save_extracted_text(pdf_contents, output_directory):
    """
    Save extracted text to organized text files.
    
    Args:
        pdf_contents (dict): Dictionary containing PDF contents
        output_directory (str): Directory to save text files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Create combined file for all texts
    combined_path = os.path.join(output_directory, "all_texts_combined.txt")
    
    with open(combined_path, 'w', encoding='utf-8') as combined_file:
        # Process each PDF's content
        for pdf_name, content in sorted(pdf_contents.items()):
            # Write individual file
            file_name = pdf_name.rsplit('.', 1)[0] + '.txt'
            individual_path = os.path.join(output_directory, file_name)
            
            with open(individual_path, 'w', encoding='utf-8') as file:
                file.write(f"=== Extracted from: {pdf_name} ===\n\n")
                file.write(content)
            
            # Add to combined file
            combined_file.write(f"\n{'='*50}\n")
            combined_file.write(f"Document: {pdf_name}\n")
            combined_file.write(f"{'='*50}\n\n")
            combined_file.write(content)
            combined_file.write("\n\n")

def main():
    # Get directory paths from user
    input_dir = input("Enter the directory path containing PDF files: ").strip()
    output_dir = input("Enter the output directory path for text files: ").strip()
    
    # Validate input directory
    if not os.path.exists(input_dir):
        print("Error: Input directory does not exist!")
        return
    
    print("\nProcessing PDF files...")
    
    # Extract text from PDFs
    pdf_contents = extract_pdf_text(input_dir)
    
    if not pdf_contents:
        print("No PDF files found in the specified directory!")
        return
    
    # Save extracted text
    save_extracted_text(pdf_contents, output_dir)
    
    print(f"\nProcessing complete!")
    print(f"- Processed {len(pdf_contents)} PDF files")
    print(f"- Individual text files saved in: {output_dir}")
    print(f"- Combined text saved as: {os.path.join(output_dir, 'all_texts_combined.txt')}")

if __name__ == "__main__":
    main()