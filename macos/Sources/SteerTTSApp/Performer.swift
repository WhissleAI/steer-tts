import Foundation
import SteerEngine
import SteerText

/// How a document is performed, sentence by sentence (the Android Narrator's render loop, shared here by the narrator
/// and the audiobook export): the reader's voice and style, headings slower, dialogue in a second voice and the line's
/// own mood, characters in voices of their own (Cast), italics stressed. Runs on the engine's queue.
struct Performer {
    let doc: Doc
    let base: Style
    let pace: Float
    let voice: String?
    let other: String?
    let cast: Cast?
    let female: [String], male: [String]

    init(doc: Doc, style: Style, speed: Float, dialogueVoices: Bool, voices: Voices, engine: Engine) {
        self.doc = doc
        base = style.copy()
        base.contextual = false   // the document's own moods apply, not the situation
        pace = speed * base.pace
        voice = engine.resolve(voice: base.voice)
        other = dialogueVoices ? Performer.dialogueVoice(for: voice, voices: voices) : voice
        cast = dialogueVoices ? doc.cast() : nil   // characters' own voices come with the dialogue voice setting
        var f: [String] = [], m: [String] = []
        for v in voices.all() { if v.gender == "male" { m.append(v.id) } else { f.append(v.id) } }
        female = f
        male = m
    }

    /// Sentence i as audio (24 kHz mono), with the pauses its segments ask for; nil if the engine was cancelled.
    func render(_ i: Int, with engine: Engine) throws -> [Float]? {
        let line = cast != nil && i < doc.paraOf.count ? cast!.lines[doc.paraOf[i]] : nil
        let speaking = line?.speaker != nil ? cast!.voiceFor(line!.speaker, narrator: voice, female: female, male: male) : other
        var audio: [Float] = []
        for seg in Narration.segments(doc.sentences[i], line) {
            let st = Performer.moodStyle(base, seg.mood)
            guard let y = try engine.synthesize(SteerText.Expression.bmp(seg.text), voice: seg.dialogue ? speaking : voice,
                                                tag: st.tagOrNull(), ctrl: st.ctrl(), pace: pace, markup: true) else { return nil }
            audio += y
            if seg.pauseAfterMs > 0 { audio += [Float](repeating: 0, count: Engine.sampleRate * seg.pauseAfterMs / 1000) }
        }
        return audio.isEmpty ? [Float](repeating: 0, count: Engine.sampleRate / 10) : audio
    }

    /// A second voice for dialogue: a featured voice of the other gender.
    static func dialogueVoice(for narrator: String?, voices: Voices) -> String? {
        let n = narrator.flatMap { voices.voice($0) }
        let want = n?.gender == "female" ? "male" : "female"
        return voices.featured().first { $0.gender == want && $0.id != narrator }?.id ?? narrator
    }

    /// The base style with a mood on top (its tag, unless the user whispers; its control shifts added).
    static func moodStyle(_ base: Style, _ m: SteerText.Expression.Mood) -> Style {
        let s = base.copy()
        if m.neutral { return s }
        if let t = m.tag, base.tag != "whisper" { s.tag = t }
        for (k, v) in m.pairs { s.add(k, v) }
        return s
    }
}
