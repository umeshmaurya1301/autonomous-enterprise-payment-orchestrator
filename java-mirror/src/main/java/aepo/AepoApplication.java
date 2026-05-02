package aepo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Entry point — Java mirror of {@code server/app.py}'s {@code main()}.
 *
 * <p>Run with:  {@code mvn spring-boot:run}  or  {@code java -jar aepo-java-mirror-*.jar}
 * <br>Defaults to port 7860 to match the HF Spaces convention used by the Python server.
 */
@SpringBootApplication
public class AepoApplication {
    public static void main(String[] args) {
        SpringApplication.run(AepoApplication.class, args);
    }
}
