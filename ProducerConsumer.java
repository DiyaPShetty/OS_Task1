public class ProducerConsumer {

    static final int SIZE = 4;
    static final int TOTAL = 12;

    static int[] buffer = new int[SIZE];
    static int in = 0;
    static int out = 0;
    static int count = 0;

    static final Object lock = new Object();

    static void produce(int value) throws InterruptedException {

        synchronized (lock) {

            while (count == SIZE) {
                System.out.println("Buffer full. Producer waiting...");
                lock.wait();
            }

            buffer[in] = value;
            in = (in + 1) % SIZE;
            count++;

            System.out.println(
                "Produced " + value + " -> " + showBuffer()
            );

            lock.notifyAll();
        }
    }

    static void consume() throws InterruptedException {

        synchronized (lock) {

            while (count == 0) {
                System.out.println("Buffer empty. Consumer waiting...");
                lock.wait();
            }

            int value = buffer[out];
            out = (out + 1) % SIZE;
            count--;

            System.out.println(
                "Consumed " + value + " -> " + showBuffer()
            );

            lock.notifyAll();
        }
    }

    static String showBuffer() {

        StringBuilder s = new StringBuilder("[");

        for (int i = 0; i < count; i++) {

            int index = (out + i) % SIZE;
            s.append(buffer[index]);

            if (i != count - 1) {
                s.append(", ");
            }
        }

        s.append("]");
        return s.toString();
    }

    public static void main(String[] args) {

        Thread producer = new Thread(() -> {

            for (int i = 1; i <= TOTAL; i++) {

                try {
                    produce(i);
                    Thread.sleep(250);

                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });

        Thread consumer = new Thread(() -> {

            for (int i = 0; i < TOTAL; i++) {

                try {
                    consume();
                    Thread.sleep(900);

                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });

        System.out.println("Producer Consumer started\n");

        consumer.start();
        producer.start();

        try {
            producer.join();
            consumer.join();

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        System.out.println("\nProducer Consumer finished");
    }
}