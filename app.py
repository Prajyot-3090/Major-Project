import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import pandas as pd
from datetime import datetime
import json
import base64
from io import BytesIO
from skimage import exposure


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics



# Example usage
# true_labels = [0, 1, 2, 0, 1, 2]
# predicted_labels = [0, 2, 1, 0, 0, 1]
# class_names = ['Normal', 'Abnormal Type 1', 'Abnormal Type 2']
# cm = generate_confusion_matrix(true_labels, predicted_labels, class_names)
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the class names
class_names = ['COVID', 'Lung_Opacity', 'No_Tumor', 'Normal', 'Tumor', 'Viral_Pneumonia']

# Color mapping for different classes (for visualization)
class_colors = {
    'COVID': '#FF5733',  # Red
    'Lung_Opacity': '#33A8FF',  # Blue
    'No_Tumor': '#33FF57',  # Green
    'Normal': '#FFFFFF',  # White
    'Tumor': '#FF33A8',  # Pink
    'Viral_Pneumonia': '#A833FF'  # Purple
}

# Create directory for storing analysis history
os.makedirs("analysis_history", exist_ok=True)

# Load the model
def load_model(model_path, num_classes):
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = models.efficientnet_b0(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    )
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

# Image preprocessing functions
def apply_preprocessing(image, contrast_adjustment=0, brightness_adjustment=0, 
                        apply_clahe=False, apply_gamma=False, gamma_value=1.0):
    """Apply various preprocessing techniques to the image"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Apply contrast adjustment
    if contrast_adjustment != 0:
        alpha = 1.0 + (contrast_adjustment / 10.0)  # Scale to reasonable range
        img_array = cv2.convertScaleAbs(img_array, alpha=alpha, beta=0)
    
    # Apply brightness adjustment
    if brightness_adjustment != 0:
        beta = brightness_adjustment * 10  # Scale to reasonable range
        img_array = cv2.convertScaleAbs(img_array, alpha=1.0, beta=beta)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if apply_clahe:
        if len(img_array.shape) > 2 and img_array.shape[2] == 3:  # Color image
            img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            img_lab = cv2.merge((cl, a_channel, b_channel))
            img_array = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        else:  # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_array = clahe.apply(img_array)
    
    # Apply gamma correction
    if apply_gamma and gamma_value != 1.0:
        img_array = exposure.adjust_gamma(img_array, gamma=gamma_value)
    
    # Convert back to PIL image
    return Image.fromarray(img_array.astype('uint8'))

# Function to generate heatmap visualization
def generate_visualization(image, predicted_class, show_heatmap=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Display the original image
    ax.imshow(image)
    
    if show_heatmap:
        # Add a semi-transparent overlay based on the predicted class color
        overlay = np.ones((*image.size[::-1], 4))  # RGBA
        color_hex = class_colors.get(predicted_class, '#FFFFFF')
        
        # Convert hex to RGB
        r = int(color_hex[1:3], 16) / 255
        g = int(color_hex[3:5], 16) / 255
        b = int(color_hex[5:7], 16) / 255
        
        overlay[:, :, 0] = r
        overlay[:, :, 1] = g
        overlay[:, :, 2] = b
        overlay[:, :, 3] = 0.3  # Alpha (transparency)
        
        ax.imshow(overlay)
    
    ax.set_title(f"Predicted: {predicted_class}")
    ax.axis('off')
    
    # Save the visualization to a temporary file
    plt.tight_layout()
    fig_path = "temp_visualization.png"
    plt.savefig(fig_path)
    plt.close(fig)
    
    return fig_path

# Function to generate comparison visualizations
def generate_comparison(original_image, processed_image, predicted_class):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the original image
    ax1.imshow(original_image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Display the processed image with overlay
    ax2.imshow(processed_image)
    
    # Add overlay
    overlay = np.ones((*processed_image.size[::-1], 4))  # RGBA
    color_hex = class_colors.get(predicted_class, '#FFFFFF')
    
    # Convert hex to RGB
    r = int(color_hex[1:3], 16) / 255
    g = int(color_hex[3:5], 16) / 255
    b = int(color_hex[5:7], 16) / 255
    
    overlay[:, :, 0] = r
    overlay[:, :, 1] = g
    overlay[:, :, 2] = b
    overlay[:, :, 3] = 0.3  # Alpha (transparency)
    
    ax2.imshow(overlay)
    ax2.set_title(f"Processed Image - {predicted_class}")
    ax2.axis('off')
    
    plt.tight_layout()
    fig_path = "temp_comparison.png"
    plt.savefig(fig_path)
    plt.close(fig)
    
    return fig_path

# Function to get recommendations based on diagnosis
def get_recommendations(condition):
    recommendations = {
        'COVID': [
            "Seek immediate medical attention if symptoms are severe.",
            "Follow isolation protocols to prevent spread.",
            "Monitor oxygen levels with a pulse oximeter if available.",
            "Stay hydrated and rest.",
            "Follow up with healthcare provider for post-COVID care."
        ],
        'Lung_Opacity': [
            "Consult with a pulmonologist for further evaluation.",
            "Additional imaging tests may be required (CT scan).",
            "Follow up to monitor changes in the opacity.",
            "Discuss treatment options based on the underlying cause."
        ],
        'No_Tumor': [
            "Regular health check-ups are still recommended.",
            "Follow standard preventive care guidelines.",
            "Report any new respiratory symptoms promptly."
        ],
        'Normal': [
            "Continue routine health maintenance.",
            "Maintain regular screening schedule based on age and risk factors.",
            "Adopt healthy lifestyle habits to maintain lung health."
        ],
        'Tumor': [
            "Urgent consultation with an oncologist is recommended.",
            "Further diagnostic tests will be needed (biopsy, PET scan).",
            "Discuss treatment options including surgery, radiation, or chemotherapy.",
            "Consider genetic testing for targeted therapies.",
            "Seek support from cancer support groups."
        ],
        'Viral_Pneumonia': [
            "Follow up with a healthcare provider for appropriate antiviral treatment.",
            "Rest and maintain good hydration.",
            "Monitor temperature and respiratory symptoms.",
            "Use prescribed medications as directed.",
            "Follow up imaging may be needed to ensure resolution."
        ]
    }
    return recommendations.get(condition, ["Please consult with a healthcare professional for specific recommendations."])

# Function to save analysis results to history
def save_to_history(image, results, predicted_class, timestamp):
    # Create a unique ID for this analysis
    analysis_id = f"{timestamp.strftime('%Y%m%d%H%M%S')}"
    
    # Save the image
    img_path = os.path.join("analysis_history", f"{analysis_id}_image.jpg")
    image.save(img_path)
    
    # Save the results
    analysis_data = {
        "id": analysis_id,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_class": predicted_class,
        "probabilities": results,
        "image_path": img_path
    }
    
    # Save as JSON
    json_path = os.path.join("analysis_history", f"{analysis_id}_data.json")
    with open(json_path, 'w') as f:
        json.dump(analysis_data, f)
    
    return analysis_id

# Function to load analysis history
def load_analysis_history():
    history_entries = []
    
    # Get all JSON files in the history directory
    json_files = [f for f in os.listdir("analysis_history") if f.endswith("_data.json")]
    
    for json_file in json_files:
        json_path = os.path.join("analysis_history", json_file)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Add to history entries
            history_entries.append({
                "id": data["id"],
                "timestamp": data["timestamp"],
                "predicted_class": data["predicted_class"],
                "image_path": data["image_path"]
            })
        except Exception as e:
            print(f"Error loading history entry {json_file}: {str(e)}")
    
    # Sort by timestamp (newest first)
    history_entries.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return history_entries

# Function to create a PDF report
def generate_report(image, results, predicted_class, description, recommendations):
    # Using an HTML template that will be converted to PDF by the browser when printing
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Image Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .image-container {{ text-align: center; margin: 20px 0; }}
            .results {{ margin: 20px 0; }}
            .recommendations {{ margin: 20px 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .footer {{ margin-top: 50px; font-size: 0.8em; color: #777; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Medical Image Analysis Report</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="image-container">
            <img src="data:image/jpeg;base64,{image_to_base64(image)}" alt="Medical Image" style="max-width: 400px;">
        </div>
        
        <div class="results">
            <h2>Analysis Results</h2>
            <p><strong>Predicted Condition:</strong> {predicted_class}</p>
            <p><strong>Description:</strong> {description}</p>
            
            <h3>Probability Distribution</h3>
            <table>
                <tr>
                    <th>Condition</th>
                    <th>Probability</th>
                </tr>
    """
    
    # Add rows for each class probability
    for class_name, prob in results.items():
        html += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{prob:.2%}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="recommendations">
            <h2>Recommendations</h2>
            <ul>
    """
    
    # Add recommendations
    for rec in recommendations:
        html += f"<li>{rec}</li>\n"
    
    html += """
            </ul>
        </div>
        
        <div class="disclaimer">
            <h2>Disclaimer</h2>
            <p>This analysis is provided for educational and informational purposes only. It should not be considered as medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional regarding any medical conditions or concerns.</p>
        </div>
        
        <div class="footer">
            <p>Powered by Medical Image Classification System</p>
        </div>
        
        <script>
            // Add a print button that will only show in the report
            document.addEventListener('DOMContentLoaded', function() {
                var printButton = document.createElement('button');
                printButton.innerHTML = 'Print Report';
                printButton.style.padding = '10px 20px';
                printButton.style.backgroundColor = '#4CAF50';
                printButton.style.color = 'white';
                printButton.style.border = 'none';
                printButton.style.borderRadius = '4px';
                printButton.style.cursor = 'pointer';
                printButton.style.display = 'block';
                printButton.style.margin = '20px auto';
                
                printButton.onclick = function() {
                    window.print();
                };
                
                document.body.appendChild(printButton);
            });
        </script>
    </body>
    </html>
    """
    
    return html

# Helper function to convert image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Function to provide descriptions of the conditions
def get_condition_description(condition):
    descriptions = {
        'COVID': "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. In chest X-rays, it typically appears as bilateral ground-glass opacities, mainly in the lower and peripheral lung fields. These opacities may be accompanied by consolidation in more severe cases.",
        'Lung_Opacity': "Lung opacity refers to any area that preferentially attenuates the X-ray beam and appears more opaque on the radiograph. This can be caused by various conditions including pneumonia, edema, hemorrhage, or atelectasis. The pattern, distribution, and associated findings help determine the underlying cause.",
        'No_Tumor': "The image shows no evidence of tumorous growth in the lungs. The lung fields appear clear without any nodular lesions or masses that would suggest neoplastic disease.",
        'Normal': "The chest X-ray appears normal with no visible abnormalities. The lung fields are clear, heart size is normal, mediastinum is unremarkable, and no pleural effusions or pneumothorax is seen. Bony structures also appear normal.",
        'Tumor': "A tumor is an abnormal growth of cells that may be benign or malignant. On X-rays, lung tumors typically appear as masses or nodules that may be solitary or multiple. Depending on the type and stage, tumors may show various characteristics including speculated margins, cavitation, or associated lymphadenopathy.",
        'Viral_Pneumonia': "Viral pneumonia is an infection of the lung caused by viruses. On chest X-rays, it often appears as interstitial patterns or ground-glass opacities that are typically bilateral and diffuse. The pattern is often reticular or reticulonodular with peribronchial thickening. Viral pneumonia generally has less consolidation compared to bacterial pneumonia."
    }
    return descriptions.get(condition, "No description available.")

# Function to make predictions
def predict(image, contrast_adjustment=0, brightness_adjustment=0, 
           apply_clahe=False, apply_gamma=False, gamma_value=1.0, 
           show_visualization=True, save_history=True):
    if image is None:
        return {class_name: 0.0 for class_name in class_names}, "No image provided", None, None, None, None, None, None
    
    start_time = time.time()
    timestamp = datetime.now()
    
    try:
        # Keep a copy of the original image
        original_image = image.copy()
        
        # Apply preprocessing if requested
        processed_image = apply_preprocessing(
            image,
            contrast_adjustment=contrast_adjustment,
            brightness_adjustment=brightness_adjustment,
            apply_clahe=apply_clahe,
            apply_gamma=apply_gamma,
            
            gamma_value=gamma_value
        )
        
        # Prepare image for model
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image_tensor = preprocess(processed_image).to(device)
        input_batch = image_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_names[predicted_idx.item()]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Generate visualizations
        visualization = None
        comparison = None
        if show_visualization:
            visualization = generate_visualization(processed_image, predicted_label)
            comparison = generate_comparison(original_image, processed_image, predicted_label)
        
        # Get condition description and recommendations
        description = get_condition_description(predicted_label)
        recommendations = get_recommendations(predicted_label)
        
        # Create the result dictionary with percentages
        result_dict = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        # Generate report
        report = generate_report(processed_image, result_dict, predicted_label, description, recommendations)
        
        # Save to history if requested
        if save_history:
            save_to_history(processed_image, result_dict, predicted_label, timestamp)
        
        return (
            result_dict, 
            predicted_label, 
            f"{processing_time:.2f} seconds", 
            description, 
            visualization, 
            comparison, 
            report,
            "\n".join(recommendations)
        )
    
    except Exception as e:
        return (
            {class_name: 0.0 for class_name in class_names}, 
            f"Error: {str(e)}", 
            None, 
            None, 
            None, 
            None, 
            None,
            None
        )

# Function to create a downloadable results file
def create_downloadable_results(result_dict, predicted_label, description, recommendations):
    # Create a CSV with the results
    csv_output = "Condition,Probability\n"
    for class_name, prob in result_dict.items():
        csv_output += f"{class_name},{prob:.4f}\n"
    
    # Add the prediction and description
    csv_output += f"\nPredicted Condition: {predicted_label}\n"
    csv_output += f"\nDescription:\n{description}\n"
    
    # Add recommendations
    csv_output += "\nRecommendations:\n"
    for i, rec in enumerate(recommendations.split('\n')):
        if rec.strip():
            csv_output += f"{i+1}. {rec}\n"
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lung_analysis_results_{timestamp}.csv"
    
    # Save to file
    with open(filename, "w") as f:
        f.write(csv_output)
    
    return filename

# Initialize the model
try:
    model_path = 'C:/Users/Pmhat/OneDrive/Desktop/Prajot-ka-project-2--main/transfer_balanced_learning_model.pth'
    num_classes = len(class_names)
    model = load_model(model_path, num_classes)
    model_status = "Model loaded successfully"
except Exception as e:
    model = None
    model_status = f"Failed to load model: {str(e)}"
    print(model_status)

# Create the Gradio interface
def main():
    # Define state variables for storing results
    current_results = {"result_dict": None, "predicted_label": None, "description": None, "recommendations": None, "image": None}
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏥 Medical Imaging Diagnostics System")
        gr.Markdown("Upload a medical image to classify lung conditions including COVID-19, Pneumonia, and Tumors")
        
        if model is None:
            gr.Markdown(f"**Warning**: {model_status}")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input Image")
                
                with gr.Accordion("Image Preprocessing", open=False):
                    contrast_slider = gr.Slider(-5, 5, value=0, step=0.5, label="Contrast Adjustment")
                    brightness_slider = gr.Slider(-5, 5, value=0, step=0.5, label="Brightness Adjustment")
                    apply_clahe = gr.Checkbox(label="Apply CLAHE (Adaptive Histogram Equalization)", value=False)
                    apply_gamma = gr.Checkbox(label="Apply Gamma Correction", value=False)
                    gamma_slider = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Gamma Value")
                
                show_viz = gr.Checkbox(label="Show Visualization", value=True)
                save_history_checkbox = gr.Checkbox(label="Save to History", value=True)
                
                submit_btn = gr.Button("Analyze Image", variant="primary")
                clear_btn = gr.Button("Clear")
                
                with gr.Accordion("Sample Images", open=False):
                    gr.Markdown("Select a sample image to analyze:")
                    with gr.Row():
                        sample_btn1 = gr.Button("COVID-19 Sample")
                        sample_btn2 = gr.Button("Normal Lung Sample")
                        sample_btn3 = gr.Button("Tumor Sample")
            
            with gr.Column(scale=2):
                with gr.Tabs() as tabs:
                    results_tab = gr.TabItem("Results")
                    with results_tab:
                        with gr.Row():
                            with gr.Column():
                                label_output = gr.Label(label="Class Probabilities")
                                prediction_output = gr.Textbox(label="Predicted Class")
                                processing_time = gr.Textbox(label="Processing Time")
                            
                            with gr.Column():
                                condition_description = gr.Textbox(label="Condition Description", lines=5)
                                
                        with gr.Accordion("Recommendations", open=True):
                            recommendations_output = gr.Textbox(label="Recommendations", lines=4)
                            
                        with gr.Row():
                            visualization_output = gr.Image(label="Visualization", show_label=True)
                            comparison_output = gr.Image(label="Original vs. Processed", show_label=True)
                        
                        gr.Markdown("### Export Options")
                        with gr.Row():
                            report_btn = gr.Button("Generate Report")
                            download_btn = gr.Button("Download Results")
                            download_output = gr.File(label="Download Results", visible=False)
                    
                    report_tab = gr.TabItem("Report")
                    with report_tab:
                        report_html = gr.HTML(label="Analysis Report")
                    
                    history_tab = gr.TabItem("History")
                    with history_tab:
                        refresh_history_btn = gr.Button("Refresh History")
                        history_dataframe = gr.Dataframe(
                            headers=["Date", "Time", "Diagnosis"],
                            datatype=["str", "str", "str"],
                            row_count=10,
                            col_count=(3, "fixed"),
                            label="Analysis History"
                        )
                        
                        selected_history_image = gr.Image(label="Selected Analysis Image")
                    
                    help_tab = gr.TabItem("Help")
                    with help_tab:
                        gr.Markdown("""
                        ## How to Use This Tool
                        
                        1. Upload a chest X-ray or lung CT scan image using the upload button
                        2. Optional: Adjust image preprocessing settings:
                           - Contrast adjustment: Increase or decrease image contrast
                           - Brightness adjustment: Make the image brighter or darker
                           - CLAHE: Apply Contrast Limited Adaptive Histogram Equalization for better feature visibility
                           - Gamma correction: Adjust the gamma value to enhance dark or bright regions
                        3. Click "Analyze Image" to run the classification
                        4. View the results in various tabs
                        
                        ## Interpreting Results
                        
                        The model provides probabilities for each possible class. Higher probability indicates greater confidence in the classification. This tool is intended for educational purposes only and should not replace professional medical advice.
                        
                        ## About the Model
                        
                        This tool uses an EfficientNet B0 model trained on a dataset of chest X-rays and CT scans. The model can classify images into six categories:
                        - COVID-19
                        - Lung Opacity (non-COVID)
                        - No Tumor
                        - Normal
                        - Tumor
                        - Viral Pneumonia
                        
                        ## Disclaimer
                        
                        This application is for educational and research purposes only. It is not intended to be a medical device and should not be used for diagnosis, treatment, or prevention of disease. Always consult with a qualified healthcare provider for medical advice.
                        """)
        
        # Define event handlers for buttons
        def process_and_store_results(results, pred_label, proc_time, desc, viz, comp, report, recs, img):
            """Store results for later use with download and report buttons"""
            if img is not None and pred_label != "No image provided" and not pred_label.startswith("Error:"):
                # Store the results
                current_results["result_dict"] = results
                current_results["predicted_label"] = pred_label
                current_results["description"] = desc
                current_results["recommendations"] = recs
                current_results["image"] = img
                current_results["report"] = report
            
            return results, pred_label, proc_time, desc, viz, comp, report, recs
        
        # Main prediction handler
        def handle_predict(*args):
            results = predict(*args)
            # Store the results for later use
            processed_results = process_and_store_results(*results, args[0])
            return processed_results
        
        submit_btn.click(
            fn=handle_predict,
            inputs=[
                input_image, 
                contrast_slider, 
                brightness_slider, 
                apply_clahe, 
                apply_gamma, 
                gamma_slider, 
                show_viz,
                save_history_checkbox
            ],
            outputs=[
                label_output, 
                prediction_output, 
                processing_time, 
                condition_description, 
                visualization_output, 
                comparison_output,
                report_html,
                recommendations_output
            ]
        )
        
        clear_btn.click(
            fn=lambda: [None, 0, 0, False, False, 1.0, True, True, None, None, None, None, None, None, None, None],
            inputs=None,
            outputs=[
                input_image, 
                contrast_slider, 
                brightness_slider, 
                apply_clahe, 
                apply_gamma, 
                gamma_slider, 
                show_viz,
                save_history_checkbox,
                label_output, 
                prediction_output, 
                processing_time, 
                condition_description, 
                visualization_output, 
                comparison_output,
                report_html,
                recommendations_output
            ]
        )
        
        # Sample image buttons
        def load_sample_image(sample_type):
            # These paths should be adjusted to where your sample images are stored
            sample_paths = {
                "covid": "sample_images/covid_sample.jpg",
                "normal": "sample_images/normal_sample.jpg",
                "tumor": "sample_images/tumor_sample.jpg"
            }
            
            path = sample_paths.get(sample_type, None)
            if path and os.path.exists(path):
                return Image.open(path)
            
            # Return a dummy image if the sample isn't found
            return None
        
        sample_btn1.click(fn=lambda: load_sample_image("covid"), outputs=input_image)
        sample_btn2.click(fn=lambda: load_sample_image("normal"), outputs=input_image)
        sample_btn3.click(fn=lambda: load_sample_image("tumor"), outputs=input_image)
        
        # History functionality
        def update_history():
            entries = load_analysis_history()
            if not entries:
                return [], None
            
            # Format for dataframe
            data = []
            for entry in entries:
                timestamp = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                data.append([
                    timestamp.strftime("%Y-%m-%d"),
                    timestamp.strftime("%H:%M:%S"),
                    entry["predicted_class"]
                ])
            
            # Load the most recent image
            latest_img = Image.open(entries[0]["image_path"]) if entries else None
            
            return data, latest_img
        
        refresh_history_btn.click(
            fn=update_history,
            outputs=[history_dataframe, selected_history_image]
        )
        
        # Report button handler - Switch to the report tab and display the latest report
        def handle_report_btn():
            if current_results["report"] is not None:
                return current_results["report"], gr.update(selected=1)  # Select Report tab (index 1)
            else:
                return "No analysis results available. Please analyze an image first.", gr.update(selected=1)
        
        report_btn.click(
            fn=handle_report_btn,
            outputs=[report_html, tabs]
        )
        
        # Download button handler
        def handle_download():
            if current_results["result_dict"] is not None and current_results["predicted_label"] is not None:
                filename = create_downloadable_results(
                    current_results["result_dict"],
                    current_results["predicted_label"],
                    current_results["description"],
                    current_results["recommendations"]
                )
                return gr.update(value=filename, visible=True)
            else:
                return gr.update(visible=False)
        
        download_btn.click(
            fn=handle_download,
            outputs=download_output
        )
        
        # Load history on startup
        demo.load(fn=update_history, outputs=[history_dataframe, selected_history_image])
        
        # Return the demo object
        return demo

# Run the app
if __name__ == "__main__":
    demo = main()
    demo.launch()
