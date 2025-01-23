import shutil
import tempfile
from datetime import date
from pathlib import Path
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from pn_domain import *  # Import your pipeline functions
from multilabeldomainclassification import *  # Import PDF processing functions
from extract_sentences import *  # Extract sentences from story
from domain_cross1 import * # Domain cross validation 

# Directory to store files persistently during session
persistent_dir = Path("persistent_files")
persistent_dir.mkdir(exist_ok=True)

app_ui = ui.page_fixed(
    ui.h2("Domain Analysis and Cross-Validation"),
    ui.markdown("Upload a PDF file to generate domain predictions."),
    ui.row(
        ui.column(
            4,
            ui.input_file("file_pdf", "Choose PDF File", accept=[".pdf"], multiple=True),
            ui.input_action_button("process_pdf", "Process PDF", class_="btn-primary"),
            ui.download_button("download_domain_predictions", "Download Domain Prediction Results"),
            ui.download_button("download_wellbeing", "Download Wellbeing Results"),
            ui.download_button("download_functioning", "Download Functioning Results"),
            ui.download_button("download_ps", "Download Problems & Symptoms Results"),
            ui.download_button("download_risk", "Download Risk Results"),
            ui.h1("Cross-Validate With Assessment Results"),
            ui.input_file("file_excel", "Upload the assessment results", accept=[".xlsx", ".csv"], multiple=False),
            ui.input_action_button("cross_validate", "Cross-Validate", class_="btn-primary"),
        ),
        ui.column(
            8,
            ui.h5("Domain Scores Table"),
            ui.output_data_frame("domain_scores_table"),  # Display domain scores without title argument
            ui.br(),
            ui.h5("Domain Predictions Table"),
            ui.output_data_frame("domain_predictions_table"),  # Display domain predictions without title argument
            ui.br(),
            ui.h5("Wellbeing Results Table"),
            ui.output_data_frame("wellbeing_table"),  # Display Wellbeing results without title argument
            ui.br(),
            ui.h5("Functioning Results Table"),
            ui.output_data_frame("functioning_table"),  # Display Functioning results without title argument
            ui.br(),
            ui.h5("Problems & Symptoms Results Table"),
            ui.output_data_frame("ps_table"),  # Display Problems & Symptoms results without title argument
            ui.br(),
            ui.h5("Risk Results Table"),
            ui.output_data_frame("risk_table"),  # Display Risk results without title argument
            ui.br(),
            ui.h5("\nCross-Validation Results Table"),
            ui.output_data_frame("cross_validation_table"),  # Display Cross-Validation results without title argument
            ui.br(),
            ui.h5("Summary Table"),
            ui.output_data_frame("summary_table"),  # Display Summary without title argument
        ),
    ),
)

def server(input: Inputs, output: Outputs, session: Session):
    # Process PDF function
    @reactive.Calc
    def process_pdf():
        if input.file_pdf() is not None:
            pdf_file_path = Path(input.file_pdf()[0]["datapath"])  # Uploaded PDF path            
            # Step 1: Process the PDF file and save the output to a persistent path
            # output_excel_path = persistent_dir / "cleaned_sentences_output.xlsx"
            # input_file = process_pdf_and_save(pdf_file_path, output_excel_path)
            input_file_df = process_pdf_and_return_dataframe(pdf_file_path.parent)

        
            # Step 2: Process Excel for predictions and pipeline
            model_names = ['f', 'w', 'ps', 'r']  # Model names
            domain_predictions_df = process_excel(input_file_df, model_names)
            
            domain_predictions_path = persistent_dir / "domain_predictions.xlsx"
            domain_predictions_df.to_excel(domain_predictions_path, index=False)

            # Step 3: Run the pipeline to get domain scores and save results
            wellbeing_df, functioning_df, ps_df, risk_df, domain_scores_df = run_pipeline(domain_predictions_path)

            # Save each result to persistent files for download
            wellbeing_path = persistent_dir / "wellbeing_pn_results.xlsx"
            functioning_path = persistent_dir / "functioning_pn_results.xlsx"
            ps_path = persistent_dir / "ps_pn_results.xlsx"
            risk_path = persistent_dir / "risk_pn_results.xlsx"
            domain_score_df_path = persistent_dir / "domain_scores.xlsx"

            domain_scores_df.to_excel(domain_score_df_path, index=False)
            wellbeing_df.to_excel(wellbeing_path, index=False)
            functioning_df.to_excel(functioning_path, index=False)
            ps_df.to_excel(ps_path, index=False)
            risk_df.to_excel(risk_path, index=False)

            # Return paths and dataframes for rendering and downloads
            return {
                "domain_predictions": domain_predictions_path,
                "wellbeing": wellbeing_path,
                "functioning": functioning_path,
                "ps": ps_path,
                "risk": risk_path,
                "domain_predictions_df": domain_predictions_df,
                "wellbeing_df": wellbeing_df,
                "functioning_df": functioning_df,
                "ps_df": ps_df,
                "risk_df": risk_df,
                "domain_scores_df": domain_scores_df,
            }
        else:
            return None

    # Display data frames in the UI
    @output
    @render.data_frame
    def domain_scores_table():
        data = process_pdf()
        if data is not None:
            # ui.h4("Domain scores table")
            return render.DataGrid(
            data["domain_scores_df"], selection_mode="none", height='fit-content', summary=True)
        
    @output
    @render.data_frame
    def domain_predictions_table():
        data = process_pdf()
        if data is not None:
            return render.DataGrid(data["domain_predictions_df"], selection_mode="none", 
                                   height='500px', summary=True)
        
    @output
    @render.data_frame
    def wellbeing_table():
        data = process_pdf()
        if data is not None:
            return render.DataGrid(data["wellbeing_df"], selection_mode="none", 
                                   height="500px", summary=True)

    @output
    @render.data_frame
    def functioning_table():
        data = process_pdf()
        if data is not None:
            return render.DataGrid(data["functioning_df"], selection_mode="none", 
                                   height="500px", summary=True)

    @output
    @render.data_frame
    def ps_table():
        data = process_pdf()
        if data is not None:
            return render.DataGrid(data["ps_df"], selection_mode="none", 
                                   height="500px", summary=True)

    @output
    @render.data_frame
    def risk_table():
        data = process_pdf()
        if data is not None:
            return render.DataGrid(data["risk_df"], selection_mode="none", 
                                   height="500px", summary=True)

    # Define cross-validation function
    @reactive.Calc
    def cross_validation():
        if input.file_excel() is not None:
            excel_file_path = Path(input.file_excel()[0]["datapath"])  # Uploaded Excel file path
            data = process_pdf()  # Get domain scores from PDF processing
            if data is not None:
                domain_scores_df = data["domain_scores_df"]
                # Perform cross-validation with the uploaded assessment results
                cross_validation_results = perform_cross_validation(excel_file_path, domain_scores_df)
                cross_validation_results_df = cross_validation_results["results_df"]
                summary_df = cross_validation_results["summary_df"]
                cross_validation_results_df["Authenticity"] = cross_validation_results_df["Authenticity"].astype(str)

                
                cross_validation_df_path = persistent_dir / "cross_validation_results.xlsx"
                cross_validation_results_df.to_excel(cross_validation_df_path, index=False)
                summary_df_path = persistent_dir / "summary_results.xlsx"
                summary_df.to_excel(summary_df_path, index=False)

                return cross_validation_df_path, cross_validation_results_df, summary_df
        return None
        
    @output
    @render.data_frame
    def cross_validation_table():
        data = cross_validation()
        if data is not None:
            # display_df = data[1][["Domain", "Score", "Assessment Score", "Difference", "Authenticity"]]
            # print(display_df)  # Verify what is actually rendered
            return render.DataGrid(data[1], selection_mode="none", 
                                   height='fit-content', summary=True)
        
        
    @render.data_frame
    def summary_table():
        data = cross_validation()
        if data is not None:
            return render.DataGrid(data[2], selection_mode="none",height='fit-content', summary=True)  # Return the DataFrame

    @render.download(filename="cross_validation_results.xlsx")
    def download_cross_validation():
        data = cross_validation()
        if data is not None:
            yield data[0].read_bytes()  # Yield the path to the cross-validation file

    # Handle file downloads
    @render.download(filename="domain_predictions.xlsx")
    def download_domain_predictions():
        data = process_pdf()
        if data is not None:
            yield data["domain_predictions"].read_bytes()

    @render.download(filename="wellbeing_pn_results.xlsx")
    def download_wellbeing():
        data = process_pdf()
        if data is not None:
            yield data["wellbeing"].read_bytes()

    @render.download(filename="functioning_pn_results.xlsx")
    def download_functioning():
        data = process_pdf()
        if data is not None:
            yield data["functioning"].read_bytes()

    @render.download(filename="ps_pn_results.xlsx")
    def download_ps():
        data = process_pdf()
        if data is not None:
            yield data["ps"].read_bytes()

    @render.download(filename="risk_pn_results.xlsx")
    def download_risk():
        data = process_pdf()
        if data is not None:
            yield data["risk"].read_bytes()

app = App(app_ui, server)
