from typing import List, Any

def filter_distinct_items(db_conn, table: str, unique_field: str, items: List[Any]) -> List[Any]:
	"""
	Filter out items that already exist in the database based on a unique field.

	Args:
		db_conn: An open database connection (e.g., psycopg2 connection).
		table (str): The table name to check against.
		unique_field (str): The field/column name to check for uniqueness.
		items (List[Any]): List of items (dict or object) to check.
	Returns:
		List[Any]: List of items that are not present in the database (distinct/new).
	"""
	if not items:
		return []

	# Extract unique values from items
	unique_values = [item[unique_field] if isinstance(item, dict) else getattr(item, unique_field) for item in items]
	
	# Prepare query
	placeholders = ','.join(['%s'] * len(unique_values))
	query = f"SELECT {unique_field} FROM {table} WHERE {unique_field} IN ({placeholders})"
 
	try:
		cursor = db_conn.cursor()
		cursor.execute(query, unique_values)
		existing = set(row[0] for row in cursor.fetchall())
		cursor.close()
	except Exception as e:
		print(f"Error checking distinct items: {e}")
		return items  # Fallback: return all as new

	# Filter out items that already exist
	new_items = [item for item in items if (item[unique_field] if isinstance(item, dict) else getattr(item, unique_field)) not in existing]
	return new_items
