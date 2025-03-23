import json
from src.ai.services.text2frame_services.elastic_service import ESEngine


es: ESEngine = ESEngine()
es.upload_to_es(
    mapping_path="/Users/trHien/DoAnTotNghiep/ViSTAR/src/init_data/modal_data.csv",
    data_path="/Users/trHien/Python/DeafEar/deafear/src/models/model_utils/manipulate/data_convert"
)