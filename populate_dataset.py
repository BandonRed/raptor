from langsmith import Client
from typing import Dict, Any
import openai
import json
import fitz

qa_pairs = [
    {
        "input": {
            "Question": "What is the main focus of Chapter 13 in the Medicare Claims Processing Manual?"
        },
        "output": {
            "message": "Chapter 13 focuses on Radiology Services and Other Diagnostic Procedures."
        }
    },
    {
        "input": {
            "Question": "Which section of Chapter 13 covers ICD Coding for Diagnostic Tests?"
        },
        "output": {
            "message": "The section titled '10 - ICD Coding for Diagnostic Tests' provides coding guidelines for diagnostic tests in radiology."
        }
    },
    {
        "input": {
            "Question": "What is the difference between the Professional Component (PC) and the Technical Component (TC) in radiology billing?"
        },
        "output": {
            "message": "The Professional Component (PC) refers to the interpretation and reporting by the radiologist, while the Technical Component (TC) covers the use of equipment and the execution of the imaging procedure."
        }
    },
    {
        "input": {
            "Question": "What information is provided under 'Payment Conditions for Radiology Services' in Chapter 13?"
        },
        "output": {
            "message": "It outlines the billing rules, payment conditions, and reimbursement guidelines for radiology services, including details for both professional and technical components."
        }
    },
    {
        "input": {
            "Question": "How are radiology services provided in leased departments treated under Medicare billing rules?"
        },
        "output": {
            "message": "Services furnished in leased hospital departments are subject to special billing instructions that differentiate them from in-house hospital services, affecting both reimbursement and claim processing."
        }
    },
    {
        "input": {
            "Question": "What is the purpose of the 'Special Rule to Incentivize Transition from Traditional X-Ray Imaging to Digital Radiography'?"
        },
        "output": {
            "message": "This rule is designed to encourage the adoption of digital radiography by providing financial incentives, thereby promoting more modern imaging technologies over traditional film-based methods."
        }
    },
    {
        "input": {
            "Question": "How does Medicare handle billing for radiology services that are not furnished in hospitals?"
        },
        "output": {
            "message": "Radiology services provided outside of hospitals are billed under the Medicare Physician Fee Schedule with specific guidelines that account for the different service settings."
        }
    },
    {
        "input": {
            "Question": "What HCPCS codes are designated for Low Osmolar Contrast Media (LOCM) in Chapter 13?"
        },
        "output": {
            "message": "The HCPCS codes for LOCM are Q9945 through Q9951."
        }
    },
    {
        "input": {
            "Question": "What payment criteria are outlined for Low Osmolar Contrast Media (LOCM) in Chapter 13?"
        },
        "output": {
            "message": "Payment for LOCM is based on clinical criteria, proper use of contrast media, and may include reductions if non-compliant imaging equipment is used."
        }
    },
    {
        "input": {
            "Question": "What are the key billing instructions for Magnetic Resonance Imaging (MRI) Procedures in this chapter?"
        },
        "output": {
            "message": "The billing instructions for MRI include specific coding requirements, the use of appropriate modifiers, and guidelines for differentiating between technical and professional components."
        }
    },
    {
        "input": {
            "Question": "How does Magnetic Resonance Angiography (MRA) differ from standard MRI as outlined in Chapter 13?"
        },
        "output": {
            "message": "MRA is specifically focused on imaging blood vessels and requires additional coding and coverage criteria compared to standard MRI, which covers a broader range of imaging procedures."
        }
    },
    {
        "input": {
            "Question": "What is the significance of modifier codes such as KX and FX in radiology billing?"
        },
        "output": {
            "message": "Modifiers like KX and FX indicate special billing circumstances or adjustments, such as confirming that a service was performed under specific conditions, which in turn can affect the payment amount."
        }
    },
    {
        "input": {
            "Question": "How does Chapter 13 address the anti-markup payment limitation for diagnostic tests?"
        },
        "output": {
            "message": "It sets forth rules that limit payment to the lowest of several amounts—such as the provider's net charge or the fee schedule amount—ensuring that excessive markups are not applied."
        }
    },
    {
        "input": {
            "Question": "What are the potential consequences of not including required line item details in a radiology claim?"
        },
        "output": {
            "message": "Failure to include necessary details like revenue codes, service dates, or proper modifiers can lead to claim denials, payment reductions, or delays in processing."
        }
    },
    {
        "input": {
            "Question": "How are Payment Conditions for PET scans addressed in Chapter 13?"
        },
        "output": {
            "message": "The chapter provides detailed billing and coverage guidelines for PET scans, specifying the appropriate CPT/HCPCS codes, modifiers, and conditions under which PET scans are reimbursed."
        }
    },
    {
        "input": {
            "Question": "What special billing instructions are provided for PET scans related to conditions such as myocardial viability or breast cancer?"
        },
        "output": {
            "message": "Special instructions include using designated CPT codes and modifiers, along with specific coverage criteria that ensure the PET scan is justified for conditions like myocardial viability or breast cancer."
        }
    },
    {
        "input": {
            "Question": "How does Chapter 13 differentiate between diagnostic and therapeutic nuclear medicine services?"
        },
        "output": {
            "message": "Diagnostic nuclear medicine services are billed separately based on the imaging procedure and radiopharmaceuticals used, whereas therapeutic nuclear medicine services may be bundled with other treatment costs and have their own coding requirements."
        }
    },
    {
        "input": {
            "Question": "Explain the use of CPT Modifier '-51' in the context of nuclear medicine billing."
        },
        "output": {
            "message": "CPT Modifier '-51' is applied when multiple procedures are performed on the same day, indicating that a reduction should be applied to the overall payment as per the multiple procedure policy."
        }
    },
    {
        "input": {
            "Question": "What role do Medicare Summary Notices (MSN) play in the radiology claims process as described in Chapter 13?"
        },
        "output": {
            "message": "MSNs provide detailed feedback on submitted claims, including explanations for denials or adjustments, and help both providers and contractors reconcile billing discrepancies."
        }
    },
    {
        "input": {
            "Question": "Analyze how Chapter 13 integrates billing instructions across various modalities (CT, MRI, PET, nuclear medicine) to create a cohesive framework for radiology services."
        },
        "output": {
            "message": "Chapter 13 unifies diverse billing rules and coding standards by offering a comprehensive set of guidelines that span multiple imaging modalities. This integration ensures consistency in claim processing, proper reimbursement, and compliance with Medicare regulations across different types of radiology services."
        }
    }
]

inputs = [qa["input"] for qa in qa_pairs]
outputs = [qa["output"] for qa in qa_pairs]


# dataset_name = "rules-engine-qa"
try:
    # uncomment to add examples into the dataset
    # client.create_examples(inputs=inputs, outputs=outputs, dataset_id="b8292a5b-b4e6-46a4-92fa-5db6d5b0e780")
    client = Client()
    
    # Get all examples from a dataset
    dataset_id = "b8292a5b-b4e6-46a4-92fa-5db6d5b0e780"  # Replace with your actual dataset ID
    examples = client.list_examples(dataset_id=dataset_id)
    
    # Extract just the questions to check for duplicates
    questions = []
    for example in examples:
        input_data = example.inputs
        if isinstance(input_data, dict) and "Question" in input_data:
            questions.append(input_data["Question"])
    
    # Check for duplicates
    from collections import Counter
    question_counts = Counter(questions)
    duplicates = {q: count for q, count in question_counts.items() if count > 1}
    
    if duplicates:
        print("Found duplicate questions:")
        for question, count in duplicates.items():
            print(f"'{question}' appears {count} times")
    else:
        print("No duplicates found")
except Exception as e:
    print("Something broke?")
    print(e)