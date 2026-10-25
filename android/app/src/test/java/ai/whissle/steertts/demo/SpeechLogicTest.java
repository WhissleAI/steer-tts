package ai.whissle.steertts.demo;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Test;

/** What the assistant and the reader say, and how it should sound: the plain-Java core behind every voice. */
public class SpeechLogicTest {
    // ---------------------------------------------------------------- moods
    @Test public void moodsFromWords() {
        assertEquals("sorry", Expression.of("I'm so sorry, the flight got cancelled").name);
        assertEquals("excited", Expression.of("Congrats!! You got the job").name);
        assertEquals("amused", Expression.of("haha that's hilarious").name);
        assertEquals("sad", Expression.of("Grandpa passed away this morning").name);
        assertEquals("urgent", Expression.of("Call me ASAP, it's urgent").name);
        assertEquals("grateful", Expression.of("Thank you so much for dinner").name);
        assertEquals("loving", Expression.of("love you, see you soon").name);
        assertEquals("annoyed", Expression.of("ugh, the train is late again").name);
        assertEquals("worried", Expression.of("Are you okay? Haven't heard from you").name);
        assertEquals("neutral", Expression.of("The meeting moved to room 4").name);
    }

    @Test public void ordinaryMessagesStayNeutral() {
        for (String t : new String[] {"Can't talk right now", "My phone died, will call later", "Great, another delay",
                "Finally home, exhausted", "Ships immediately", "That is not great news", "The meeting moved to room 4",
                "Done with the report", "Sorry I'm late, parking", "Perfect"})
            assertTrue(t + " -> " + Expression.of(t).name, Expression.of(t).tag == null || Expression.of(t).name.equals("sorry") && t.startsWith("Sorry I'm"));
        assertEquals("annoyed", Expression.of("seriously? again").name);
    }

    @Test public void groupMessagesKeepTheirSenders() {
        List<String[]> msgs = Arrays.asList(new String[] {"Sam", "On my way"}, new String[] {"Sam", "10 minutes"},
                new String[] {"Mom", "Drive safe ❤️"});
        Speech sp = Readout.conversation(msgs, "Family", "WhatsApp", true, null);
        assertEquals("Sam in Family on WhatsApp.", sp.parts.get(0).text);
        assertEquals("On my way. 10 minutes.", sp.parts.get(1).text);
        assertEquals("Then Mom.", sp.parts.get(2).text);
        assertEquals("Drive safe.", sp.parts.get(3).text);
        assertEquals("loving", sp.parts.get(3).mood.name);
        assertNotEquals(sp.parts.get(1).voice, sp.parts.get(3).voice);     // Mom is not read in Sam's voice
    }

    @Test public void moodsFromEmojiPunctuationAndCapitals() {
        assertEquals("amused", Expression.of("did you see that 😂😂").name);
        assertEquals("sad", Expression.of("missed the bus 😭").name);
        assertEquals("excited", Expression.of("We're here!!!").name);
        Expression.Mood shout = Expression.of("HAPPY BIRTHDAY!!! 🎉🎂");
        assertEquals("excited", shout.name);
        assertTrue(shout.ctrl.get("loud") > Expression.EXCITED.ctrl.get("loud"));   // capitals add loudness
        assertEquals("annoyed", Expression.of("WHY IS THIS STILL BROKEN 😡").name);
    }

    @Test public void questionsRiseUnlessUrgent() {
        Expression.Mood q = Expression.of("Are we still on for four?");
        assertTrue(q.ctrl.get("f0_end") > 1);
        Expression.Mood q2 = Expression.of("Can you come? 🙂");                     // a trailing emoji still ends a question
        assertTrue(q2.ctrl.containsKey("f0_end"));
        assertFalse(Expression.of("Emergency, can you call me now?").ctrl.containsKey("f0_end"));
    }

    @Test public void moodsCarryTagAndControls() {
        for (Expression.Mood m : new Expression.Mood[] {Expression.SAD, Expression.EXCITED, Expression.HAPPY, Expression.SORRY,
                Expression.URGENT, Expression.AMUSED, Expression.ANNOYED}) {
            assertTrue(m.name, m.tag != null && !m.tag.isEmpty());
            assertFalse(m.name + ": a tag alone is barely audible on the edge model", m.ctrl.isEmpty());
        }
    }

    // ---------------------------------------------------------------- chat text into words
    @Test public void speakableChatText() {
        assertEquals("I don't know, be right back", Expression.speakable("idk, brb 😅"));
        assertEquals("I don't know, be right back.", Readout.sentence("idk, brb 😅"));
        assertEquals("that was soo funny", Expression.speakable("that was soooooo funny lol"));
        assertEquals("see you tomorrow", Expression.speakable("see u tmrw 👋"));
        assertEquals("a thumbs up", Expression.describeEmoji("👍"));
        assertEquals("a heart", Expression.describeEmoji("❤️"));
        assertNull(Expression.describeEmoji("ok 👍"));
        assertFalse(Expression.bmp("hi 🎉").contains("🎉"));   // never reaches JNI as broken UTF-8
    }

    // ---------------------------------------------------------------- readouts
    @Test public void readoutFramesThenMessageInItsMoodAndVoice() {
        Speech sp = Readout.message("Sam", null, "WhatsApp", Collections.singletonList("I'm so sorry, the flight got cancelled"), true, null);
        assertEquals(2, sp.parts.size());
        assertEquals("Sam on WhatsApp.", sp.parts.get(0).text);
        assertTrue(sp.parts.get(0).mood.neutral());
        assertNull(sp.parts.get(0).voice);                                  // the assistant's own voice
        assertEquals("I'm so sorry, the flight got cancelled.", sp.parts.get(1).text);
        assertEquals("sorry", sp.parts.get(1).mood.name);
        assertNotEquals(null, sp.parts.get(1).voice);                       // Sam's voice
    }

    @Test public void personVoicesAreStableDistinctAndNeverTheUsers() {
        String sam = Voices.forPerson("Sam", null), sam2 = Voices.forPerson("  sam ", null);
        assertEquals(sam, sam2);
        assertNotEquals(Voices.FEATURED[0], sam);                            // not the default (the user's) voice
        assertNotEquals(Voices.FEATURED[1], Voices.forPerson("Mom", Voices.FEATURED[1]));
        long distinct = Arrays.asList("Sam", "Mom", "Alex", "Priya", "Jo", "Chris").stream().map(n -> Voices.forPerson(n, null)).distinct().count();
        assertTrue("six people should not all share a voice", distinct >= 3);
    }

    @Test public void emojiOnlyMessageIsDescribed() {
        Speech sp = Readout.message("Mom", null, "Messages", Collections.singletonList("👍"), true, null);
        assertEquals(1, sp.parts.size());
        assertEquals("Mom sent a thumbs up.", sp.parts.get(0).text);
    }

    @Test public void groupAndLongMessages() {
        Speech sp = Readout.message("Jo", "Family", "WhatsApp", Collections.singletonList("Dinner at 7?"), false, null);
        assertEquals("Jo in Family on WhatsApp.", sp.parts.get(0).text);
        assertNull(sp.parts.get(1).voice);                                   // person voices off
        StringBuilder longText = new StringBuilder();
        for (int i = 0; i < 80; ++i) longText.append("word ");
        Speech l = Readout.message("Jo", null, "Mail", Collections.singletonList(longText.toString()), false, null);
        assertTrue(l.parts.get(1).text.endsWith(", and more."));
        assertTrue(l.parts.get(1).text.length() <= Readout.MAX_CHARS + 12);
    }

    @Test public void callsEscalate() {
        long t = 1_000_000;
        assertEquals(1, Readout.ringCount("Dana", t));
        assertEquals(2, Readout.ringCount("dana", t + 60_000));
        assertEquals(3, Readout.ringCount("Dana", t + 120_000));
        assertEquals(1, Readout.ringCount("Dana", t + 120_000 + 16 * 60_000));   // a quarter of an hour later: a new call
        assertEquals("Incoming call from Dana.", Readout.call("Dana", null, 1).text());
        assertEquals("Dana is calling again.", Readout.call("Dana", null, 2).text());
        Speech third = Readout.call("Dana", null, 3);
        assertTrue(third.text().contains("third time") && third.text().contains("urgent"));
        assertEquals("urgent", third.mainMood().name);
        assertEquals("Incoming WhatsApp call from Dana.", Readout.call("Dana", "WhatsApp", 1).text());
    }

    @Test public void moreThingsWorthHearing() {
        assertEquals("You missed a call from Mom.", Readout.missedCall("Mom").text());
        assertEquals("Reminder: Call the dentist.", Readout.reminder("Call the dentist", null, "Tasks").text());
        assertEquals("Timer: Time's up.", Readout.reminder("Timer", "Time's up", "Clock").text());
        assertEquals("Uber. Your driver is arriving. Toyota Camry, ABC 123.",
                Readout.update("Your driver is arriving", "Toyota Camry, ABC 123", "Uber").text());
        assertEquals("Email from Priya. Contract signed.", Readout.email("Priya", "Contract signed").text());
        assertEquals("Battery at 15 percent. Time to charge.", Readout.battery(15, false).text());
        assertEquals("Your phone is fully charged.", Readout.battery(100, true).text());
    }

    // ---------------------------------------------------------------- catch-up
    @Test public void digestGroupsByPerson() {
        Digest.clear();
        Digest.add("k1", "Sam", "WhatsApp", "running late");
        Digest.add("k2", "Mom", "Messages", "call me tonight ❤️");
        Digest.add("k3", "Sam", "WhatsApp", "10 minutes away");
        assertEquals(3, Digest.size());
        Digest.removeKey("k2");                                              // seen on screen: not read again
        Speech sp = Digest.take(false, null);
        assertTrue(sp.text(), sp.text().startsWith("You have 2 new messages, from Sam."));
        assertTrue(sp.text().contains("running late. 10 minutes away."));
        assertEquals(0, Digest.size());
        assertTrue(Digest.take(false, null).text().contains("all caught up"));
        assertEquals("Sam, Mom and Alex", Digest.names(Arrays.asList("Sam", "Mom", "Alex")));
        assertEquals("Sam, Mom and 3 others", Digest.names(Arrays.asList("Sam", "Mom", "A", "B", "C")));
    }

    // ---------------------------------------------------------------- books
    @Test public void dialogueApartFromNarration() {
        List<Narration.Seg> s = Narration.segments("“Run! They're coming!” she shouted, and grabbed the bag.");
        assertEquals(2, s.size());
        assertTrue(s.get(0).dialogue);
        assertEquals("excited", s.get(0).mood.name);
        assertFalse(s.get(1).dialogue);
        assertEquals("she shouted, and grabbed the bag.", s.get(1).text);
    }

    @Test public void scareQuotesStayNarration() {
        List<Narration.Seg> s = Narration.segments("The so-called “expert” was wrong.");
        assertEquals(1, s.size());
        assertFalse(s.get(0).dialogue);
    }

    @Test public void emphasisAndHeadings() {
        String em = "It was " + Narration.EM_ON + "never" + Narration.EM_OFF + " about the money.";
        assertEquals("It was *never* about the money.", Narration.segments(em).get(0).text);
        assertEquals("It was never about the money.", Narration.display(em));
        String broken = "He said " + Narration.EM_ON + "maybe.";                     // a sentence split inside italics
        assertEquals("He said maybe.", Narration.segments(broken).get(0).text);
        List<Narration.Seg> h = Narration.segments(Narration.HEAD + "Chapter 2");
        assertTrue(h.get(0).heading);
        assertTrue(h.get(0).pauseAfterMs >= 500);
        assertEquals("Chapter 2", Narration.display(Narration.HEAD + "Chapter 2"));
        assertEquals("a (laugh) b", Narration.markup("a [laugh] b"));             // not a sound event from a book
    }

    @Test public void sentencesKeepMarkersAndAbbreviations() {
        List<String> s = TextClean.sentences(Collections.singletonList(
                "Dr. Smith arrived. " + Narration.EM_ON + "Nobody" + Narration.EM_OFF + " saw him. Then he left."));
        assertEquals(3, s.size());
        assertEquals("Dr. Smith arrived.", s.get(0));
        assertEquals("Nobody saw him.", Narration.display(s.get(1)));
    }

    @Test public void dialogueIsNotSplitInsideQuotes() {
        List<String> s = TextClean.sentences(Collections.singletonList("“Run! They're coming!” she shouted. It was quiet."));
        assertEquals(2, s.size());
        assertEquals("“Run! They're coming!” she shouted.", s.get(0));
        assertTrue(Narration.segments(s.get(0)).get(0).dialogue);
        List<String> st = TextClean.sentences(Collections.singletonList("\"Stop. Now.\" He left."));
        assertEquals(2, st.size());
    }

    // ---------------------------------------------------------------- screens: summaries of what reads badly
    @Test public void summariesOfTablesAndLists() {
        List<List<String>> info = Arrays.asList(Arrays.asList("Author", "Lewis Carroll"), Arrays.asList("Illustrator", "John Tenniel"),
                Arrays.asList("Genre", "Children's fiction"), Arrays.asList("Publisher", "Macmillan"), Arrays.asList("Pages", "192"),
                Arrays.asList("Followed by", "Through the Looking-Glass"));
        assertEquals("Details: Author, Lewis Carroll. Illustrator, John Tenniel. Genre, Children's fiction. Publisher, Macmillan. And 2 more.",
                Summary.phrase(info, 2));
        List<List<String>> data = Arrays.asList(Arrays.asList("Region", "Q2", "Q3", "Change"), Arrays.asList("North", "$1.1M", "$1.3M", "+18%"),
                Arrays.asList("South", "$0.9M", "$1.0M", "+11%"));
        assertEquals("A table of 2 rows, with columns Region, Q2, Q3 and Change. The first row: North, $1.1M, $1.3M, +18%.", Summary.phrase(data, 4));
        List<List<String>> links = new java.util.ArrayList<>();
        for (String x : new String[] {"Alice", "Mad Hatter", "Cheshire Cat", "Queen of Hearts", "White Rabbit", "Dormouse", "Caterpillar"})
            links.add(Collections.singletonList(x));
        assertEquals("A list of 7 items, starting with Alice, Mad Hatter and Cheshire Cat.", Summary.phrase(links, 1));
        List<List<String>> refs = Arrays.asList(Collections.singletonList("^ Smith (2001). Retrieved 3 May 2020."),
                Collections.singletonList("^ Jones. doi:10.1/x"), Collections.singletonList("^ Brown, p. 12"));
        assertEquals("A list of 3 references, skipped.", Summary.phrase(refs, 1));
        assertNull(Summary.phrase(Arrays.asList(Collections.singletonList("Preheat the oven to 200 degrees and grease the tin well before you start."),
                Collections.singletonList("Mix the flour, sugar and butter until it looks like breadcrumbs, then add the eggs.")), 1));   // read it
    }

    // ---------------------------------------------------------------- stories
    @Test public void castFindsSpeakersAndDelivery() {
        List<String> p = Arrays.asList(
                "Alice walked into the garden.",
                "“Who are you?” said the Caterpillar.",
                "“I hardly know, sir,” Alice replied, rather shyly.",
                "“Explain yourself!” he shouted.",
                "“I can’t,” she whispered.",
                "“Why not?”",
                "“Because I’m not myself, you see.”");
        Cast c = Cast.analyze(p);
        assertNull(c.lines[0].speaker);
        assertEquals("Caterpillar", c.lines[1].speaker);
        assertEquals("asked", c.lines[1].delivery == null ? "asked" : c.lines[1].delivery.name);
        assertEquals("Alice", c.lines[2].speaker);
        assertEquals("female", c.lines[2].gender);
        assertEquals("Caterpillar", c.lines[3].speaker);                     // "he": the last man (a caterpillar is "he")
        assertEquals("shouted", c.lines[3].delivery.name);
        assertEquals("Alice", c.lines[4].speaker);                           // "she"
        assertEquals("whispered", c.lines[4].delivery.name);
        assertEquals("Caterpillar", c.lines[5].speaker);                     // no names: they take turns
        assertEquals("Alice", c.lines[6].speaker);
    }

    @Test public void quotesOfEveryKind() {
        assertTrue(Cast.hasDialogue("‘Now, my dears,’ said old Mrs. Rabbit, ‘you may go.’"));
        assertTrue(Cast.hasDialogue("'Now my dears,' said old Mrs. Rabbit."));
        assertFalse(Cast.hasDialogue("Peter didn't go into Mr. McGregor's garden."));    // apostrophes, not quotes
        assertEquals("Mrs. Rabbit", Cast.analyze(Arrays.asList("'Now my dears,' said old Mrs. Rabbit one morning.")).lines[0].speaker);
        List<String> s = TextClean.sentences(Collections.singletonList("‘Run! They’re coming!’ she cried. Nobody moved."));
        assertEquals(2, s.size());
        Narration.Seg d = Narration.segments(s.get(0)).get(0);
        assertTrue(d.dialogue);
        assertEquals("Run! They’re coming!", d.text);
    }

    @Test public void aliasesAreOneCharacter() {
        Cast c = Cast.analyze(Arrays.asList("“Oh dear!” said the White Rabbit.", "“Late!” said the Rabbit’s voice.", "“I’m late,” said the Rabbit."));
        assertEquals("White Rabbit", c.lines[0].speaker);
        assertEquals("White Rabbit", c.lines[2].speaker);
        List<String> f = Arrays.asList("voice_f1", "voice_f2", "voice_f3"), m = Arrays.asList("voice_m1", "voice_m2", "voice_m3");
        assertEquals(c.voiceFor("White Rabbit", "voice_f1", f, m), c.voiceFor("White Rabbit", "voice_f1", f, m));
        assertNotEquals("voice_f1", c.voiceFor("White Rabbit", "voice_f1", f, m));    // never the narrator's voice
    }

    @Test public void sceneMoodIsLightAndNeedsTwoCues() {
        assertEquals("neutral", Cast.scene("It was dark.").name);
        assertEquals("tense", Cast.scene("In the darkness she heard footsteps, and a cold shiver ran down her back.").name);
        assertEquals("sombre", Cast.scene("At the funeral there were tears; she felt so alone.").name);
    }

    @Test public void htmlKeepsEmphasisAndHeadings() {
        String t = Extract.htmlToText("<h1>One</h1><p>It was <em>cold</em>.</p>");
        assertTrue(t.contains(Narration.HEAD + "One"));
        assertTrue(t.contains(Narration.EM_ON + "cold" + Narration.EM_OFF));
    }

    // ---------------------------------------------------------------- books from the web (Gutenberg EPUB markup)
    @Test public void bookMarkupReadsCleanly() {
        String html = "<h2>LIST OF ILLUSTRATIONS.</h2><table>"
                + "<tr><td>“<span class=\"smcap\">He gathered herbs</span>”</td><td><a href=\"b.html#Page_213\">213</a></td></tr>"
                + "<tr><td>A Gleam of Sunshine</td><td><a href=\"b.html#Page_249\">249</a></td></tr></table>"
                + "<p><a href=\"c.html#ch1\">CHAPTER I. Down the Rabbit-Hole</a></p>"
                + "<p class=\"figcenter\"><span title=\"The Custom House\" id=\"img_images_illu_001.jpg\">The Custom House</span></p>"
                + "<div class=\"initial\"><span title=\"I\" id=\"img_images_initial_i.jpg\">I</span></div>"
                + "<p><span class=\"firstwords\">t</span> is a little remarkable, said “<span>he</span>.”</p>";
        String text = Extract.htmlToText(html);
        assertFalse(text, text.contains("249") || text.contains("Gleam") || text.contains("Rabbit-Hole") || text.contains("Custom House"));
        assertTrue(text, text.contains("It is a little remarkable, said “he.”"));
    }
}
