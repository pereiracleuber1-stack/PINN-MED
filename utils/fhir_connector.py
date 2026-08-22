import json
import datetime

class FHIRHospitalConnector:
    """
    Mecanismo de Ingestão e Exportação Padrão HL7 / FHIR R4
    Compatível com prontuários eletrônicos Epic Systems, Cerner e Philips Tasy.
    """
    @staticmethod
    def parse_patient_bundle(fhir_bundle_json):
        """Lê JSON FHIR R4 e converte em tensores fisiológicos normalizados."""
        data = json.loads(fhir_bundle_json) if isinstance(fhir_bundle_json, str) else fhir_bundle_json
        
        extracted = {
            "patient_id": data.get("id", "UNKNOWN-PATIENT"),
            "age": 60,
            "sofa_score": 4,
            "lactate_baseline": 2.0,
            "comorbidities_count": 0,
            "map_baseline": 80.0
        }
        
        # Varredura de Recursos do Bundle FHIR
        for entry in data.get("entry", []):
            resource = entry.get("resource", {})
            r_type = resource.get("resourceType")
            
            if r_type == "Patient":
                birth_date = resource.get("birthDate", "1965-01-01")
                try:
                    birth_year = int(birth_date.split("-")[0])
                    extracted["age"] = max(18, 2026 - birth_year)
                except:
                    extracted["age"] = 60
                    
            elif r_type == "Observation":
                code = resource.get("code", {}).get("coding", [{}])[0].get("code", "")
                val = resource.get("valueQuantity", {}).get("value", None)
                if code == "25107-6" and val: # LOINC Lactato
                    extracted["lactate_baseline"] = float(val)
                elif code == "8478-0" and val: # LOINC Pressão Arterial Média
                    extracted["map_baseline"] = float(val)
                elif code == "SOFA-SCORE" and val:
                    extracted["sofa_score"] = int(val)
                    
            elif r_type == "Condition":
                extracted["comorbidities_count"] += 1
                
        return extracted

    @staticmethod
    def export_prediction_to_fhir(patient_id, peak_lactate, shock_hour, sha256_hash):
        """Gera recurso FHIR Observation estruturado para retorno ao EHR."""
        return {
            "resourceType": "Observation",
            "id": f"SGP-PRED-{patient_id}",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "exam",
                    "display": "Exam"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://sgp-pinn.health/codes",
                    "code": "SEPTIC-SHOCK-TRAJECTORY",
                    "display": "SGP-PINN Septic Shock AI Trajectory Prediction"
                }]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.datetime.utcnow().isoformat() + "Z",
            "component": [
                {
                    "code": {"text": "Projected Peak Lactate (mmol/L)"},
                    "valueQuantity": {"value": round(peak_lactate, 2), "unit": "mmol/L"}
                },
                {
                    "code": {"text": "Predicted Shock Transition Time (Hours)"},
                    "valueQuantity": {"value": round(shock_hour, 1), "unit": "hours"}
                }
            ],
            "identifier": [{
                "system": "urn:ietf:rfc:3986",
                "value": f"urn:sha256:{sha256_hash}"
            }]
        }
