import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint import SimpleCheckpoint

def validate_data(df: pd.DataFrame, expectation_suite_name: str):
    """Валидация данных с помощью Great Expectations."""
    context = ge.get_context()
    
    # Создаем batch из DataFrame
    batch = ge.from_pandas(df)
    
    # Загружаем сьют ожиданий
    batch = context.get_expectation_suite(expectation_suite_name)
    
    # Запускаем валидацию
    results = context.run_validation_operator(
        "action_list_operator",
        assets_to_validate=[batch],
        run_id=f"validation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    if not results["success"]:
        raise ValueError("Валидация данных провалилась! Проверьте данные на аномалии.")
    else:
        print("Валидация данных прошла успешно!")