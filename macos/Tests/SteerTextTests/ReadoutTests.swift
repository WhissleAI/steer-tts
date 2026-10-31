import XCTest
@testable import SteerText

/// What the assistant says for messages, calls, emails and catch-up. Ported from the Android app's SpeechLogicTest (the
/// tests that touch Readout, Speech and Digest), plus Mac tests for the Digest's key file and clock.
final class ReadoutTests: XCTestCase {
    // ---------------------------------------------------------------- chat text into words
    func testSentenceFromChatText() {
        XCTAssertEqual("I don't know, be right back.", Readout.sentence("idk, brb 😅"))
    }

    // ---------------------------------------------------------------- readouts
    func testGroupMessagesKeepTheirSenders() {
        let msgs = [("Sam", "On my way"), ("Sam", "10 minutes"), ("Mom", "Drive safe ❤️")]
        let sp = Readout.conversation(msgs, "Family", "WhatsApp", true, nil)
        XCTAssertEqual("Sam in Family on WhatsApp.", sp.parts[0].text)
        XCTAssertEqual("On my way. 10 minutes.", sp.parts[1].text)
        XCTAssertEqual("Then Mom.", sp.parts[2].text)
        XCTAssertEqual("Drive safe.", sp.parts[3].text)
        XCTAssertEqual("loving", sp.parts[3].mood.name)
        XCTAssertNotEqual(sp.parts[1].voice, sp.parts[3].voice)     // Mom is not read in Sam's voice
    }

    func testReadoutFramesThenMessageInItsMoodAndVoice() {
        let sp = Readout.message("Sam", nil, "WhatsApp", ["I'm so sorry, the flight got cancelled"], true, nil)
        XCTAssertEqual(2, sp.parts.count)
        XCTAssertEqual("Sam on WhatsApp.", sp.parts[0].text)
        XCTAssertTrue(sp.parts[0].mood.neutral)
        XCTAssertNil(sp.parts[0].voice)                                  // the assistant's own voice
        XCTAssertEqual("I'm so sorry, the flight got cancelled.", sp.parts[1].text)
        XCTAssertEqual("sorry", sp.parts[1].mood.name)
        XCTAssertNotNil(sp.parts[1].voice)                               // Sam's voice
    }

    func testEmojiOnlyMessageIsDescribed() {
        let sp = Readout.message("Mom", nil, "Messages", ["👍"], true, nil)
        XCTAssertEqual(1, sp.parts.count)
        XCTAssertEqual("Mom sent a thumbs up.", sp.parts[0].text)
    }

    func testGroupAndLongMessages() {
        let sp = Readout.message("Jo", "Family", "WhatsApp", ["Dinner at 7?"], false, nil)
        XCTAssertEqual("Jo in Family on WhatsApp.", sp.parts[0].text)
        XCTAssertNil(sp.parts[1].voice)                                   // person voices off
        let longText = String(repeating: "word ", count: 80)
        let l = Readout.message("Jo", nil, "Mail", [longText], false, nil)
        XCTAssertTrue(l.parts[1].text.hasSuffix(", and more."))
        XCTAssertTrue(l.parts[1].text.jlength <= Readout.MAX_CHARS + 12)
    }

    func testCallsEscalate() {
        let t: Int64 = 1_000_000
        XCTAssertEqual(1, Readout.ringCount("Dana", t))
        XCTAssertEqual(2, Readout.ringCount("dana", t + 60_000))
        XCTAssertEqual(3, Readout.ringCount("Dana", t + 120_000))
        XCTAssertEqual(1, Readout.ringCount("Dana", t + 120_000 + 16 * 60_000))   // a quarter of an hour later: a new call
        XCTAssertEqual("Incoming call from Dana.", Readout.call("Dana", nil, 1).text())
        XCTAssertEqual("Dana is calling again.", Readout.call("Dana", nil, 2).text())
        let third = Readout.call("Dana", nil, 3)
        XCTAssertTrue(third.text().contains("third time") && third.text().contains("urgent"))
        XCTAssertEqual("urgent", third.mainMood().name)
        XCTAssertEqual("Incoming WhatsApp call from Dana.", Readout.call("Dana", "WhatsApp", 1).text())
    }

    func testMoreThingsWorthHearing() {
        XCTAssertEqual("You missed a call from Mom.", Readout.missedCall("Mom").text())
        XCTAssertEqual("Reminder: Call the dentist.", Readout.reminder("Call the dentist", nil, "Tasks").text())
        XCTAssertEqual("Timer: Time's up.", Readout.reminder("Timer", "Time's up", "Clock").text())
        XCTAssertEqual("Uber. Your driver is arriving. Toyota Camry, ABC 123.",
                       Readout.update("Your driver is arriving", "Toyota Camry, ABC 123", "Uber").text())
        XCTAssertEqual("Email from Priya. Contract signed.", Readout.email("Priya", "Contract signed").text())
        XCTAssertEqual("Battery at 15 percent. Time to charge.", Readout.battery(15, false).text())
        XCTAssertEqual("Your phone is fully charged.", Readout.battery(100, true).text())
    }

    func testEmailSaysWhoWhatAndHowItStarts() {
        let sp = Readout.email("Priya Sharma", "Contract signed", "Contract signed\nHi Karan, the signed copy is attached. Let me know if anything else is needed.")
        XCTAssertEqual("Email from Priya Sharma.", sp.parts[0].text)
        XCTAssertEqual("Contract signed.", sp.parts[1].text)
        XCTAssertEqual("Hi Karan, the signed copy is attached.", sp.parts[2].text)
        XCTAssertEqual("Google", Readout.senderName("no-reply@accounts.google.com"))
        XCTAssertEqual("Acme", Readout.senderName("billing@acme.co.uk"))
        XCTAssertEqual("John Smith", Readout.senderName("john.smith@gmail.com"))
        XCTAssertEqual("You have 3 new emails.", Readout.email("3 new messages", "x").parts[0].text)
        XCTAssertEqual(2, Readout.email("Sam", "Lunch?", "Lunch?").parts.count)
    }

    func testAMessageInHindiIsAnnouncedNotSilenced() {
        let sp = Readout.message("Priya", nil, "WhatsApp", ["नमस्ते, कल मिलते हैं"], false, nil)
        XCTAssertEqual("Priya sent a message on WhatsApp, in Hindi. I can't read Hindi yet.", sp.text())
        let who = Readout.message("प्रिया", nil, "WhatsApp", ["See you at 7"], false, nil)
        XCTAssertTrue(who.text().hasPrefix("Someone on WhatsApp."), who.text())
    }

    // ---------------------------------------------------------------- speech
    func testSpeechTrimsSkipsEmptyAndAddsPauses() {
        let sp = Speech.of("  Hello  ").add("   ").add(nil).pause(200).add("Sad news", Expression.SAD)
        XCTAssertEqual(2, sp.parts.count)
        XCTAssertEqual("Hello Sad news", sp.text())
        XCTAssertEqual(200, sp.parts[0].pauseAfterMs)
        XCTAssertEqual("sad", sp.mainMood().name)
        XCTAssertTrue(Speech().pause(100).isEmpty)
        XCTAssertEqual("neutral", Speech.of("plain").mainMood().name)
    }

    // ---------------------------------------------------------------- catch-up
    func testDigestGroupsByPerson() {
        let digest = Digest()
        digest.clear()
        digest.add("k1", "Sam", "WhatsApp", "running late")
        digest.add("k2", "Mom", "Messages", "call me tonight ❤️")
        digest.add("k3", "Sam", "WhatsApp", "10 minutes away")
        XCTAssertEqual(3, digest.size())
        digest.removeKey("k2")                                              // seen on screen: not read again
        let sp = digest.take(false, nil)
        XCTAssertTrue(sp.text().hasPrefix("You have 2 new messages, from Sam."), sp.text())
        XCTAssertTrue(sp.text().contains("running late. 10 minutes away."))
        XCTAssertEqual(0, digest.size())
        XCTAssertTrue(digest.take(false, nil).text().contains("all caught up"))
        XCTAssertEqual("Sam, Mom and Alex", Digest.names(["Sam", "Mom", "Alex"]))
        XCTAssertEqual("Sam, Mom and 3 others", Digest.names(["Sam", "Mom", "A", "B", "C"]))
    }

    func testDigestKeepsOnlyKeysOnDiskAndRestores() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("digest-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        var now: Int64 = 5_000_000
        let a = Digest(directory: dir, clock: { now })
        a.add("k1", "Sam", "WhatsApp", "secret words")
        a.add("k2", "Mom", "Messages", "more secret words")
        let file = try String(contentsOf: dir.appendingPathComponent("kept.keys"), encoding: .utf8)
        XCTAssertFalse(file.contains("secret"))                              // never the words
        XCTAssertTrue(file.contains("k1\t5000000"))

        now += 60_000                                                        // a restart
        let b = Digest(directory: dir, clock: { now })
        XCTAssertEqual(["k1", "k2"], b.savedKeys())
        XCTAssertEqual(0, b.size())
        b.restore("k1", "Sam", "WhatsApp", "secret words")
        b.restore("k1", "Sam", "WhatsApp", "secret words")                  // not added twice
        b.restore("k9", "Jo", "Mail", "never kept")                          // not one of ours
        b.forgetExcept(["k1"])                                               // k2's notification is gone
        XCTAssertEqual(["k1"], b.savedKeys())
        XCTAssertEqual(1, b.size())

        now += 12 * 3600_000                                                 // twelve hours after it was kept
        XCTAssertEqual(0, b.size())
        XCTAssertEqual(Set<String>(), b.savedKeys())
        XCTAssertEqual(Set<String>(), Digest(directory: dir, clock: { now }).savedKeys())
    }
}
