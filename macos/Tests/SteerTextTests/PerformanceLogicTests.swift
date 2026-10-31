import XCTest
@testable import SteerText

/// How the reader and the assistant should sound: moods, chat text, person voices, dialogue and narration, the cast of
/// a story, context styling, and the library and style files. Ported from the Android app's SpeechLogicTest (the
/// tests that touch Expression, Narration, Cast and Voices), plus storage tests for the Mac.
final class PerformanceLogicTests: XCTestCase {
    private let EM_ON = Narration.EM_ON, EM_OFF = Narration.EM_OFF, HEAD = Narration.HEAD

    // ---------------------------------------------------------------- moods
    func testMoodsFromWords() {
        XCTAssertEqual("sorry", Expression.of("I'm so sorry, the flight got cancelled").name)
        XCTAssertEqual("excited", Expression.of("Congrats!! You got the job").name)
        XCTAssertEqual("amused", Expression.of("haha that's hilarious").name)
        XCTAssertEqual("sad", Expression.of("Grandpa passed away this morning").name)
        XCTAssertEqual("urgent", Expression.of("Call me ASAP, it's urgent").name)
        XCTAssertEqual("grateful", Expression.of("Thank you so much for dinner").name)
        XCTAssertEqual("loving", Expression.of("love you, see you soon").name)
        XCTAssertEqual("annoyed", Expression.of("ugh, the train is late again").name)
        XCTAssertEqual("worried", Expression.of("Are you okay? Haven't heard from you").name)
        XCTAssertEqual("neutral", Expression.of("The meeting moved to room 4").name)
    }

    func testOrdinaryMessagesStayNeutral() {
        for t in ["Can't talk right now", "My phone died, will call later", "Great, another delay",
                  "Finally home, exhausted", "Ships immediately", "That is not great news", "The meeting moved to room 4",
                  "Done with the report", "Sorry I'm late, parking", "Perfect"] {
            let m = Expression.of(t)
            XCTAssertTrue(m.tag == nil || m.name == "sorry" && t.hasPrefix("Sorry I'm"), "\(t) -> \(m.name)")
        }
        XCTAssertEqual("annoyed", Expression.of("seriously? again").name)
    }

    func testMoodsFromEmojiPunctuationAndCapitals() {
        XCTAssertEqual("amused", Expression.of("did you see that 😂😂").name)
        XCTAssertEqual("sad", Expression.of("missed the bus 😭").name)
        XCTAssertEqual("excited", Expression.of("We're here!!!").name)
        let shout = Expression.of("HAPPY BIRTHDAY!!! 🎉🎂")
        XCTAssertEqual("excited", shout.name)
        XCTAssertGreaterThan(shout.ctrl["loud"]!, Expression.EXCITED.ctrl["loud"]!)   // capitals add loudness
        XCTAssertEqual("annoyed", Expression.of("WHY IS THIS STILL BROKEN 😡").name)
    }

    func testQuestionsRiseUnlessUrgent() {
        let q = Expression.of("Are we still on for four?")
        XCTAssertGreaterThan(q.ctrl["f0_end"]!, 1)
        let q2 = Expression.of("Can you come? 🙂")                     // a trailing emoji still ends a question
        XCTAssertNotNil(q2.ctrl["f0_end"])
        XCTAssertNil(Expression.of("Emergency, can you call me now?").ctrl["f0_end"])
    }

    func testMoodsCarryTagAndControls() {
        for m in [Expression.SAD, Expression.EXCITED, Expression.HAPPY, Expression.SORRY, Expression.URGENT,
                  Expression.AMUSED, Expression.ANNOYED] {
            XCTAssertTrue(m.tag != nil && !m.tag!.isEmpty, m.name)
            XCTAssertFalse(m.ctrl.isEmpty, m.name + ": a tag alone is barely audible on the edge model")
        }
    }

    func testMoodPlusMergesAndKeepsOrder() {
        let m = Expression.EXCITED.plus("in capitals: louder", ("loud", 0.6)).plus("question: rising end", ("f0_end", 1.4))
        XCTAssertEqual(m.ctrlKeys, ["rate", "loud", "f0_iqr", "f0_med", "f0_end"])
        XCTAssertEqual(m.ctrl["loud"]!, Float(0.9) + Float(0.6))
        XCTAssertEqual(m.why, "excited: livelier, faster; in capitals: louder; question: rising end")
        XCTAssertTrue(m.name == "excited" && m.tag == "excited")
        XCTAssertEqual(Expression.describe(Expression.NEUTRAL), "")
    }

    // ---------------------------------------------------------------- chat text into words
    func testSpeakableChatText() {
        XCTAssertEqual("I don't know, be right back", Expression.speakable("idk, brb 😅"))
        XCTAssertEqual("that was soo funny", Expression.speakable("that was soooooo funny lol"))
        XCTAssertEqual("see you tomorrow", Expression.speakable("see u tmrw 👋"))
        XCTAssertEqual("a thumbs up", Expression.describeEmoji("👍"))
        XCTAssertEqual("a heart", Expression.describeEmoji("❤️"))
        XCTAssertNil(Expression.describeEmoji("ok 👍"))
        XCTAssertFalse(Expression.bmp("hi 🎉").contains("🎉"))   // never reaches the engine as broken UTF-8
        XCTAssertEqual(Expression.describeEmoji("🦄"), "an emoji")
        XCTAssertTrue(Expression.emojiOnly(" 👍🏽 "))
    }

    // ---------------------------------------------------------------- person voices
    func testPersonVoicesAreStableDistinctAndNeverTheUsers() {
        let sam = Voices.forPerson("Sam", own: nil), sam2 = Voices.forPerson("  sam ", own: nil)
        XCTAssertEqual(sam, sam2)
        XCTAssertNotEqual(Voices.FEATURED[0], sam)                            // not the default (the user's) voice
        XCTAssertNotEqual(Voices.FEATURED[1], Voices.forPerson("Mom", own: Voices.FEATURED[1]))
        let distinct = Set(["Sam", "Mom", "Alex", "Priya", "Jo", "Chris"].map { Voices.forPerson($0, own: nil) }).count
        XCTAssertGreaterThanOrEqual(distinct, 3, "six people should not all share a voice")
    }

    func testPersonVoicesMatchJavaHash() {
        // String.hashCode by hand: "sam" = 113631 -> floorMod(113631, 7) = 0 -> the first of the pool without voice_1688
        XCTAssertEqual(Voices.forPerson("Sam", own: nil), "voice_2500")
        // a long name overflows int: the same wrap as Java's
        var h: Int32 = 0
        for u in "christopher alexander".utf16 { h = 31 &* h &+ Int32(u) }
        let pool = Array(Voices.FEATURED.dropFirst())
        XCTAssertEqual(Voices.forPerson("Christopher Alexander", own: nil), pool[Int(((h % 7) + 7) % 7)])
    }

    // ---------------------------------------------------------------- books
    func testDialogueApartFromNarration() {
        let s = Narration.segments("“Run! They're coming!” she shouted, and grabbed the bag.")
        XCTAssertEqual(2, s.count)
        XCTAssertTrue(s[0].dialogue)
        XCTAssertEqual("excited", s[0].mood.name)
        XCTAssertFalse(s[1].dialogue)
        XCTAssertEqual("she shouted, and grabbed the bag.", s[1].text)
    }

    func testScareQuotesStayNarration() {
        let s = Narration.segments("The so-called “expert” was wrong.")
        XCTAssertEqual(1, s.count)
        XCTAssertFalse(s[0].dialogue)
    }

    func testEmphasisAndHeadings() {
        let em = "It was " + EM_ON + "never" + EM_OFF + " about the money."
        XCTAssertEqual("It was *never* about the money.", Narration.segments(em)[0].text)
        XCTAssertEqual("It was never about the money.", Narration.display(em))
        let broken = "He said " + EM_ON + "maybe."                     // a sentence split inside italics
        XCTAssertEqual("He said maybe.", Narration.segments(broken)[0].text)
        let h = Narration.segments(HEAD + "Chapter 2")
        XCTAssertTrue(h[0].heading)
        XCTAssertGreaterThanOrEqual(h[0].pauseAfterMs, 500)
        XCTAssertEqual("Chapter 2", Narration.display(HEAD + "Chapter 2"))
        XCTAssertEqual("a (laugh) b", Narration.markup("a [laugh] b"))             // not a sound event from a book
    }

    func testDialogueIsNotSplitInsideQuotes() {
        let s = TextClean.sentences(["“Run! They're coming!” she shouted. It was quiet."])
        XCTAssertEqual(2, s.count)
        XCTAssertEqual("“Run! They're coming!” she shouted.", s[0])
        XCTAssertTrue(Narration.segments(s[0])[0].dialogue)
        XCTAssertEqual(2, TextClean.sentences(["\"Stop. Now.\" He left."]).count)
    }

    func testDialogueTakesTheSpeakersDelivery() {
        let c = Cast.analyze(["“I can’t do this,” she whispered."])
        let s = Narration.segments("“I can’t do this,” she whispered.", c.lines[0])
        XCTAssertEqual("whispered", s[0].mood.name)                      // a tag of its own: the delivery wins
        let asked = Cast.analyze(["“Is it far, Tom?” Alice asked."])
        let q = Narration.segments("“Is it far, Tom?” Alice asked.", asked.lines[0])
        XCTAssertNotNil(q[0].mood.ctrl["f0_end"])                        // no tag: the line's own mood plus the delivery
    }

    // ---------------------------------------------------------------- stories
    func testCastFindsSpeakersAndDelivery() {
        let p = [
            "Alice walked into the garden.",
            "“Who are you?” said the Caterpillar.",
            "“I hardly know, sir,” Alice replied, rather shyly.",
            "“Explain yourself!” he shouted.",
            "“I can’t,” she whispered.",
            "“Why not?”",
            "“Because I’m not myself, you see.”"]
        let c = Cast.analyze(p)
        XCTAssertNil(c.lines[0].speaker)
        XCTAssertEqual("Caterpillar", c.lines[1].speaker)
        XCTAssertEqual("asked", c.lines[1].delivery?.name ?? "asked")
        XCTAssertEqual("Alice", c.lines[2].speaker)
        XCTAssertEqual("female", c.lines[2].gender)
        XCTAssertEqual("Caterpillar", c.lines[3].speaker)                     // "he": the last man (a caterpillar is "he")
        XCTAssertEqual("shouted", c.lines[3].delivery?.name)
        XCTAssertEqual("Alice", c.lines[4].speaker)                           // "she"
        XCTAssertEqual("whispered", c.lines[4].delivery?.name)
        XCTAssertEqual("Caterpillar", c.lines[5].speaker)                     // no names: they take turns
        XCTAssertEqual("Alice", c.lines[6].speaker)
    }

    func testQuotesOfEveryKind() {
        XCTAssertTrue(Cast.hasDialogue("‘Now, my dears,’ said old Mrs. Rabbit, ‘you may go.’"))
        XCTAssertTrue(Cast.hasDialogue("'Now my dears,' said old Mrs. Rabbit."))
        XCTAssertFalse(Cast.hasDialogue("Peter didn't go into Mr. McGregor's garden."))    // apostrophes, not quotes
        XCTAssertEqual("Mrs. Rabbit", Cast.analyze(["'Now my dears,' said old Mrs. Rabbit one morning."]).lines[0].speaker)
        let s = TextClean.sentences(["‘Run! They’re coming!’ she cried. Nobody moved."])
        XCTAssertEqual(2, s.count)
        let d = Narration.segments(s[0])[0]
        XCTAssertTrue(d.dialogue)
        XCTAssertEqual("Run! They’re coming!", d.text)
    }

    func testAliasesAreOneCharacter() {
        let c = Cast.analyze(["“Oh dear!” said the White Rabbit.", "“Late!” said the Rabbit’s voice.", "“I’m late,” said the Rabbit."])
        XCTAssertEqual("White Rabbit", c.lines[0].speaker)
        XCTAssertEqual("White Rabbit", c.lines[2].speaker)
        let f = ["voice_f1", "voice_f2", "voice_f3"], m = ["voice_m1", "voice_m2", "voice_m3"]
        XCTAssertEqual(c.voiceFor("White Rabbit", narrator: "voice_f1", female: f, male: m),
                       c.voiceFor("White Rabbit", narrator: "voice_f1", female: f, male: m))
        XCTAssertNotEqual("voice_f1", c.voiceFor("White Rabbit", narrator: "voice_f1", female: f, male: m))    // never the narrator's voice
    }

    func testSceneMoodIsLightAndNeedsTwoCues() {
        XCTAssertEqual("neutral", Cast.scene("It was dark.").name)
        XCTAssertEqual("tense", Cast.scene("In the darkness she heard footsteps, and a cold shiver ran down her back.").name)
        XCTAssertEqual("sombre", Cast.scene("At the funeral there were tears; she felt so alone.").name)
    }

    func testNamesInNarrationAndOutside() {
        // "she" in dialogue resolves to the last woman named in narration
        // (the first word of a sentence is not taken for a name, as in Java)
        XCTAssertNil(Cast.analyze(["Mary came in.", "“Hello there,” she said."]).lines[1].speaker)
        let c = Cast.analyze(["In came Mary with Tom.", "“Hello there,” she said."])
        XCTAssertEqual("Mary", c.lines[1].speaker)
        XCTAssertEqual(Cast.outside("“Go away now,” he said, “so-called” friend."), " ¶  he said, “so-called” friend.")
        XCTAssertEqual(Cast.canonical("the Rabbit’s"), "Rabbit")
        XCTAssertEqual(Cast.genderOf("Mrs. Hudson"), "female")
    }

    // ---------------------------------------------------------------- context styling
    func testContextStylerRules() {
        let base = Style()
        let day = ContextStyler.DeviceState(hour: 12)
        let alert = ContextStyler.decide(base, text: "Warning: low battery", caller: nil, device: ContextStyler.DeviceState(hour: 23))
        XCTAssertEqual(alert.reason, "alert: faster, louder, brighter")       // alerts are not softened at night
        XCTAssertEqual(alert.z[1], 1.2, accuracy: 1e-6)
        let q = ContextStyler.decide(base, text: "Are you there?", caller: nil, device: day)
        XCTAssertEqual(q.reason, "question: rising end")
        let good = ContextStyler.decide(base, text: "Congrats on the move", caller: nil, device: day)
        XCTAssertEqual(good.reason, "good news: livelier")
        let tb = ContextStyler.decide(base, text: "Congrats button?", caller: ContextStyler.TALKBACK, device: day)
        XCTAssertEqual(tb.reason, "screen reader: brisk")                     // a screen reader's labels carry no mood
        let night = ContextStyler.decide(base, text: "Hello", caller: nil, device: ContextStyler.DeviceState(hour: 2, lowBattery: true))
        XCTAssertEqual(night.reason, "night: softer; battery low: calmer")
        let nav = ContextStyler.decide(base, text: "Turn left", caller: ContextStyler.MAPS, device: ContextStyler.DeviceState(hour: 23, quietRinger: true))
        XCTAssertEqual(nav.reason, "navigation: clear")                       // directions stay clear at night
        XCTAssertEqual(ContextStyler.decide(base, text: "Hi", caller: nil, device: ContextStyler.DeviceState(hour: 15, quietRinger: true)).reason,
                       "ringer off: softer")
        XCTAssertEqual(ContextStyler.decide(base, text: "Hi", caller: nil, device: day).reason, "neutral: your settings")
        let off = base.copy()
        off.contextual = false
        XCTAssertEqual(ContextStyler.decide(off, text: "URGENT!!", caller: nil, device: day).reason, "your settings (context styling off)")
    }

    // ---------------------------------------------------------------- style
    func testStylePresetsAndCtrl() {
        let s = Style()
        s.applyPreset("Happy")
        XCTAssertEqual(s.tag, "happiness")
        XCTAssertEqual(s.preset, "Happy")
        XCTAssertEqual(s.ctrl(), "rate=0.40,f0_med=0.70,f0_iqr=1.30,tilt=0.50")
        XCTAssertEqual(s.preset("Calm").reason, "calm: serenity + controls")
        XCTAssertEqual(s.preset("Natural").reason, "natural: controls")
        XCTAssertNil(Style().ctrl())
        XCTAssertNil(Style().tagOrNull())
        let big = Style()
        big.add("loud", 5)
        XCTAssertEqual(big.z[1], Style.LIMIT)
        let half = Style()
        half.add("rate", 0.125)
        XCTAssertEqual(half.ctrl(), "rate=0.13")                               // Java's HALF_UP, not printf's half even
    }

    func testStyleJsonRoundTrip() throws {
        let s = Style()
        s.voice = "voice_0010"
        s.applyPreset("Serious")
        s.pace = 1.3
        s.smooth = true
        s.reason = "why"
        let url = tempDir().appendingPathComponent("style.json")
        s.save(to: url)
        let o = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        XCTAssertEqual(Set(o.keys), ["voice", "tag", "z", "contextual", "pace", "preset", "smooth", "reason"])
        XCTAssertEqual((o["z"] as! [Any]).count, 12)
        let r = Style.load(from: url)
        XCTAssertEqual(r.voice, "voice_0010")
        XCTAssertEqual(r.tag, "confidence")
        XCTAssertEqual(r.z, s.z)
        XCTAssertEqual(r.pace, 1.3)
        XCTAssertTrue(r.smooth)
        XCTAssertEqual(r.preset, "Serious")
        XCTAssertEqual(r.reason, "why")
        XCTAssertTrue(r.contextual)
        // the Android file, as org.json writes it: null voice, clamped values, missing fields defaulted
        let android = #"{"voice":null,"tag":"none","z":[9,0,0,0,0,0,0,0,0,0,0,0],"contextual":false,"pace":5}"#
        let a = Style.fromJson(data: Data(android.utf8))!
        XCTAssertNil(a.voice)
        XCTAssertEqual(a.z[0], 3)
        XCTAssertEqual(a.pace, Style.MAX_PACE)
        XCTAssertFalse(a.contextual)
        XCTAssertEqual(a.preset, "Natural")
        XCTAssertEqual(Style.load(from: url.appendingPathExtension("missing")).pace, Style.DEFAULT_PACE)
    }

    // ---------------------------------------------------------------- voices
    private func voicesJSON() -> URL {
        // macos/Tests/SteerTextTests/<this file> -> repo root -> c/models/steer-tts-edge/voices.json
        URL(fileURLWithPath: #filePath).resolvingSymlinksInPath().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("c/models/steer-tts-edge/voices.json")
    }

    func testVoiceNamesAreStableFromTheModelsVoicesJson() throws {
        let url = voicesJSON()
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "no voices.json at \(url.path)")
        let v = Voices(voicesJSON: url, store: MemoryVoicesStore())
        XCTAssertFalse(v.isEmpty)
        let again = Voices(voicesJSON: url, store: MemoryVoicesStore())
        XCTAssertEqual(v.all().map { $0.name }, again.all().map { $0.name })
        // names follow voice order per gender, from the fixed lists
        let all = v.all()
        let firstFemale = all.first { $0.gender == "female" }!, firstMale = all.first { $0.gender != "female" }!
        XCTAssertEqual(firstFemale.name, "Ava")
        XCTAssertEqual(firstMale.name, "Leo")
        XCTAssertEqual(all.first!.id, "voice_0005")                           // the file's first voice: male, high, slow
        XCTAssertEqual(all.first!.describe(), "Bright, relaxed")
        XCTAssertEqual(all.first!.genderLabel, "Male")
        XCTAssertEqual(v.voice("voice_0010")?.name, "Ava")
        XCTAssertEqual(v.voice("voice_0010")?.describe(), "Warm, steady")
        // every featured voice is in the model, and the defaults have names
        XCTAssertEqual(v.featured().count, Voices.FEATURED.count)
        XCTAssertEqual(v.nameOf(nil), v.nameOf("female"))
        XCTAssertEqual(v.nameOf("female"), v.voice(Voices.FEATURED[0])!.name)
        XCTAssertEqual(v.nameOf("male"), v.voice(Voices.FEATURED[1])!.name)
        XCTAssertEqual(v.nameOf("voice_9999"), "voice_9999")
    }

    func testFavourites() {
        let store = MemoryVoicesStore()
        let v = Voices(data: Data(#"{"voices":[{"name":"voice_1","gender":"female"},{"name":"voice_2","gender":"male"}]}"#.utf8), store: store)
        XCTAssertEqual(v.all().map { $0.name }, ["Ava", "Leo"])
        XCTAssertFalse(v.isFavourite("voice_1"))
        v.toggleFavourite("voice_1")
        XCTAssertTrue(v.isFavourite("voice_1"))
        XCTAssertTrue(Voices(data: Data(), store: store).isFavourite("voice_1"))   // kept in the store
        v.toggleFavourite("voice_1")
        XCTAssertFalse(v.isFavourite("voice_1"))
        XCTAssertTrue(Voices(data: Data("nope".utf8), store: store).isEmpty)
    }

    // ---------------------------------------------------------------- library
    private func tempDir() -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("steer-perform-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return d
    }

    func testDocSaveLoadRoundTrip() throws {
        let lib = tempDir()
        let d = Doc(title: "Alice", author: "Lewis Carroll", kind: "epub", source: "https://example.org/alice.epub",
                    paragraphs: [HEAD + "Chapter 1", "Alice was tired. She sat by her sister.", "“Who are you?” said the Caterpillar."])
        XCTAssertEqual(d.sentences.count, 4)
        XCTAssertEqual(d.paraOf, [0, 1, 1, 2])
        XCTAssertEqual(d.displayTitle, "Alice · Lewis Carroll")
        d.save(in: lib)
        let files = Set(try FileManager.default.contentsOfDirectory(atPath: lib.path))
        XCTAssertEqual(files, [d.id + ".json", d.id + ".meta"])
        let o = try JSONSerialization.jsonObject(with: Data(contentsOf: lib.appendingPathComponent(d.id + ".json"))) as! [String: Any]
        for k in ["id", "title", "kind", "source", "added", "pos", "opened", "paragraphs"] { XCTAssertNotNil(o[k], k) }

        d.pos = 2
        d.opened = d.added + 5000
        d.savePos(in: lib)
        let posText = try String(contentsOf: lib.appendingPathComponent(d.id + ".pos"), encoding: .utf8)
        XCTAssertEqual(posText, "2 \(d.opened) 0")
        let r = try XCTUnwrap(Doc.load(d.id, in: lib))
        XCTAssertEqual(r.id, d.id)
        XCTAssertEqual(r.title, "Alice")
        XCTAssertEqual(r.author, "Lewis Carroll")
        XCTAssertEqual(r.kind, "epub")
        XCTAssertEqual(r.source, d.source)
        XCTAssertEqual(r.paragraphs, d.paragraphs)
        XCTAssertEqual(r.sentences, d.sentences)
        XCTAssertEqual(r.added, d.added)
        XCTAssertEqual(r.pos, 2)
        XCTAssertEqual(r.opened, d.opened)
        XCTAssertFalse(r.done)
        XCTAssertEqual(r.progress(), 0.5)
        XCTAssertEqual(r.cast().lines[2].speaker, "Caterpillar")

        let metas = Doc.list(in: lib)
        XCTAssertEqual(metas.count, 1)
        XCTAssertEqual(metas[0].pos, 2)
        XCTAssertEqual(metas[0].count, 4)
        XCTAssertEqual(metas[0].source, d.source)
        XCTAssertEqual(metas[0].chars, Int64(d.sentences.reduce(0) { $0 + $1.jlength }))
        XCTAssertEqual(metas[0].minutesLeft(1), 1)
        XCTAssertEqual(Doc.findSource(d.source, titlePrefix: nil, in: lib), d.id)
        XCTAssertNil(Doc.findSource("elsewhere", titlePrefix: nil, in: lib))

        // finished: done comes back from the position file; the reading queue leaves it out
        r.done = true
        r.pos = 0
        r.savePos(in: lib)
        XCTAssertTrue(Doc.load(d.id, in: lib)!.done)
        XCTAssertEqual(Doc.list(in: lib)[0].progress(), 1)
        XCTAssertTrue(Doc.upNext(in: lib, except: nil).isEmpty)

        Doc.delete(d.id, in: lib)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: lib.path).isEmpty)
        XCTAssertNil(Doc.load(d.id, in: lib))
    }

    func testDocReadsAnAndroidFile() throws {
        // a document as the Android app writes it (no .meta yet: list() reads it once and writes one)
        let lib = tempDir()
        let json = #"{"id":"abc","title":"Notes · Me","kind":"text","source":"","added":1700000000000,"pos":7,"opened":1700000001000,"paragraphs":["One. Two.","Three."]}"#
        try Data(json.utf8).write(to: lib.appendingPathComponent("abc.json"))
        let m = Doc.list(in: lib)
        XCTAssertEqual(m.count, 1)
        XCTAssertEqual(m[0].title, "Notes · Me")
        XCTAssertEqual(m[0].count, 3)
        XCTAssertEqual(m[0].pos, 2)                                            // clamped to the last sentence
        XCTAssertTrue(FileManager.default.fileExists(atPath: lib.appendingPathComponent("abc.meta").path))
        XCTAssertEqual(Doc.load("abc", in: lib)!.added, 1_700_000_000_000)
    }

    func testPdfDocKeepsSpansAndRegions() throws {
        let lib = tempDir()
        let p = "First sentence here. Second one."
        let spans = [[PdfLayout.Span(page: 2, start: 0, end: p.jlength, l: 0.1, t: 0.2, r: 0.9, b: 0.25)]]
        let d = Doc.createPdf(id: "pdf1", title: "Paper", source: "file:///paper.pdf", paragraphs: [p], spans: spans)
        XCTAssertEqual(d.sentences.count, 2)
        XCTAssertEqual(d.pageOf(1), 2)
        XCTAssertEqual(d.regions![0][0][1], 0.1, accuracy: 1e-6)
        d.save(in: lib)
        let r = try XCTUnwrap(Doc.load("pdf1", in: lib))
        XCTAssertEqual(r.spans?.first?.first?.page, 2)
        XCTAssertEqual(r.regions!.count, 2)
        XCTAssertEqual(r.regions![1][0][3], d.regions![1][0][3], accuracy: 1e-6)
    }

    func testSaveOrExistingDedupes() throws {
        let lib = tempDir()
        // the same source opens the copy already there, where it was left
        let book = Doc(title: "Alice", kind: "epub", source: "gutenberg:11", paragraphs: ["Once upon a time."])
        XCTAssertEqual(book.saveOrExisting(in: lib), book.id)
        let again = Doc(title: "Alice (another title)", kind: "epub", source: "gutenberg:11", paragraphs: ["Once upon a time."])
        XCTAssertEqual(again.saveOrExisting(in: lib), book.id)
        XCTAssertEqual(Doc.list(in: lib).count, 1)

        // an older copy without a source: the same kind and title is the same document
        let old = Doc(title: "Report", kind: "pdf", source: nil, paragraphs: ["Old copy."])
        old.save(in: lib)
        let fresh = Doc(title: "Report", kind: "pdf", source: "file:///report.pdf", paragraphs: ["Old copy."])
        XCTAssertEqual(fresh.saveOrExisting(in: lib), old.id)

        // the same typed text replaces its older copy (heard again from the top)
        let t1 = Doc(title: "Pasted", kind: "text", source: nil, paragraphs: ["Hello there."])
        XCTAssertEqual(t1.saveOrExisting(in: lib), t1.id)
        let t2 = Doc(title: "Pasted", kind: "text", source: nil, paragraphs: ["Hello there."])
        XCTAssertEqual(t2.saveOrExisting(in: lib), t2.id)
        XCTAssertNil(Doc.load(t1.id, in: lib))
        XCTAssertNotNil(Doc.load(t2.id, in: lib))
        // different text under the same title is a new document
        let t3 = Doc(title: "Pasted", kind: "text", source: nil, paragraphs: ["Something else entirely."])
        XCTAssertEqual(t3.saveOrExisting(in: lib), t3.id)
        XCTAssertNotNil(Doc.load(t2.id, in: lib))
        XCTAssertEqual(Doc.list(in: lib).count, 4)

        XCTAssertEqual(Doc.findSource("nowhere", titlePrefix: "Repo", in: lib), old.id)   // an older copy, by its title
        XCTAssertEqual(Doc.upNext(in: lib, except: t3.id).map { $0.id }.contains(t3.id), false)
        XCTAssertTrue(Doc.createTransient(title: "Screen", paragraphs: ["Hi."]).isTransient)
    }
}
