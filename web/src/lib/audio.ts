/**
 * Client-side audio inspection.
 *
 * Decoding the file in the browser means the duration shown to the user is the real
 * one, measured the same way the server measures it. v1 enforced its "15 second
 * minimum" as a byte count — `len(audio_bytes) < 15000`, which is about a sixth of a
 * second of CD-quality audio — so the rule it advertised was not the rule it applied.
 */

/** Decoded duration in seconds, or `null` if the browser cannot decode the file. */
export async function decodeDuration(file: Blob): Promise<number | null> {
  try {
    const buffer = await file.arrayBuffer()
    const ctx = new OfflineAudioContext(1, 1, 44_100)
    const decoded = await ctx.decodeAudioData(buffer)
    return decoded.duration
  } catch {
    // An unsupported codec is not worth a stack trace here: the server decodes it again
    // and will reject it with a proper message if it really is unreadable.
    return null
  }
}
