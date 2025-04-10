
all: v1d v2 v3 v4

v1: v1.c
	$(CC) $(CFLAGS)  -Wall -O0  -o v1 v1.c  -lm
	
v2: v2.c
	$(CC) $(CFLAGS)  -Wall -o v2 v2.c -lm

v2i: v2instrucciones.c
	$(CC) $(CFLAGS)  -Wall -o v2i v2instrucciones.c -lm
	
v3: v3.c
	$(CC) $(CFLAGS) -mavx2 -Wall  -o v3 v3.c -lm
	
v4: v4.c
	$(CC) $(CFLAGS) -fopenmp  -Wall -o v4 v4.c -lm

run: v1d v2 v3 v4
	./v1d $(ARGS)
	./v2 $(ARGS)
	./v3 $(ARGS)
	./v4 $(ARGS)

clean:
	rm -f v1d v2 v3 v4