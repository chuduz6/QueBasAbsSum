from pyrouge import Rouge155

r = Rouge155()
r.system_dir = 'D:\Better Work'
r.model_dir = 'D:\Better Work'
r.system_filename_pattern = 'system_summary.txt'
r.model_filename_pattern = 'reference_summary.txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)