import argparse

from app.rag.vector_store import get_vector_store


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", required=True)
	args = parser.parse_args()
	vs = get_vector_store()
	vs.delete_collection(args.dataset)
	print(f"Deleted collection for dataset {args.dataset}")


if __name__ == "__main__":
	main()
