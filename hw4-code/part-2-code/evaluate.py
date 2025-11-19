from argparse import ArgumentParser
from utils import compute_metrics

parser = ArgumentParser()
parser.add_argument("-ps", "--predicted_sql", dest = "pred_sql",
    required = True, help = "path to your model's predicted SQL queries")
parser.add_argument("-pr", "--predicted_records", dest = "pred_records",
    required = True, help = "path to the predicted development database records")
parser.add_argument("-ds", "--development_sql", dest = "dev_sql",
    required = True, help = "path to the ground-truth development SQL queries")
parser.add_argument("-dr", "--development_records", dest = "dev_records",
    required = True, help = "path to the ground-truth development database records")

args = parser.parse_args()

# compute_metrics returns: sql_em, record_em, record_f1, model_error_msgs
sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
    args.dev_sql, args.pred_sql, args.dev_records, args.pred_records
)

# Compute error rate
num_errors = sum(1 for msg in model_error_msgs if msg != "")
error_rate = num_errors / len(model_error_msgs) if len(model_error_msgs) > 0 else 0.0

# Print all metrics
print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"SQL Exact Match (EM):     {sql_em:.4f} ({sql_em*100:.2f}%)")
print(f"Record Exact Match (EM):  {record_em:.4f} ({record_em*100:.2f}%)")
print(f"Record F1 Score:          {record_f1:.4f} ({record_f1*100:.2f}%)")
print(f"SQL Error Rate:           {error_rate:.4f} ({error_rate*100:.2f}%)")
print(f"Total Queries:            {len(model_error_msgs)}")
print(f"Queries with SQL Errors:  {num_errors}")
print("=" * 50)