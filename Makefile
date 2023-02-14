all:	program1	program2

program1:	quasimV1.cu
	nvcc	-o	quamsimV1	quasimV1.cu

program2: quasimV2.cu
	nvcc -o quamsimV2 quasimV1.cu

clean:
	rm quamsimV1 quamsimV2

