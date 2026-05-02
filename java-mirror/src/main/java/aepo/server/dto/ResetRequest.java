package aepo.server.dto;

/**
 * POST /reset request body — Java mirror of the Python {@code {"task": ...}} payload.
 *
 * <p>{@code task} is nullable: missing field defaults to {@code "easy"}, matching
 * the Python server's behaviour. Validation of the value (must be easy/medium/hard)
 * happens in the controller so the error message is consistent with the Python's
 * 422 response.
 */
public record ResetRequest(String task, Long seed) { }
