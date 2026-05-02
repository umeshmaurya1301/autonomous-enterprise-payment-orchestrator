package aepo.server.dto;

import aepo.types.AEPOAction;

/**
 * POST /step request body — wraps an {@link AEPOAction} so the JSON shape is
 * {@code {"action": { ... } }}, matching the Python server.
 */
public record StepRequest(AEPOAction action) { }
