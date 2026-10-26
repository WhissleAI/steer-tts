import XCTest
@testable import SteerText

/// Text as it is said (Spoken): what reached the engine as digits, symbols and accents now reaches it as words.
/// Ported case for case from the Android app's SpokenTest.java.
final class SpokenTests: XCTestCase {
    override func setUp() { Spoken.dayFirst = false }

    private func say(_ input: String, _ out: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(Spoken.say(input), out, input, file: file, line: line)
    }

    func testEverydayText() {
        say("10:00 dinner", "ten o'clock dinner")
        say("Dinner at 10:00 PM", "Dinner at ten pee em")
        say("10:00 – 11:00 dinner", "ten o'clock to eleven o'clock dinner")
        say("Chapter 1. The beginning", "Chapter one. The beginning")
        say("CHAPTER I.", "CHAPTER one.")
        say("Chapter XII", "Chapter twelve")
        say("Café crème brûlée in São Paulo with Zoë and Dvořák", "Cafe creme brulee in Sao Paulo with Zoe and Dvorak")
        say("naïve façade, résumé, Pokémon", "naive facade, resume, Pokemon")
        say("It was 2015 and 10,000 people came.", "It was twenty fifteen and ten thousand people came.")
        say("Call me at 9:30am on Mon, Oct 5", "Call me at nine thirty ay em on Monday, October fifth")
        say("Meeting 2:30–3:15 pm", "Meeting two thirty to three fifteen pee em")
        say("Order #12345 arrives Tue 3/14", "Order number one two three, four five arrives Tuesday, March fourteenth")
        say("$1,299.99 or €20 or ₹500", "one thousand two hundred ninety nine dollars and ninety nine cents or twenty euros or five hundred rupees")
        say("The 1990s; the '80s", "The nineteen nineties; the eighties")
        say("Version 2.3.1, 4G LTE, 5km, 72°F, 50%", "Version two point three point one, four G LTE, five kilometers, seventy two degrees Fahrenheit, fifty percent")
        say("Mr. Smith lives at 221B Baker St.", "Mr. Smith lives at two hundred twenty one B Baker Street.")
        say("I. Down the Rabbit-Hole", "One. Down the Rabbit-Hole")
        say("II. The Pool of Tears", "Two. The Pool of Tears")
        say("Rs. 450, 3 hrs, 2 mins", "four hundred fifty rupees, three hours, two minutes")
        say("5 × 3 = 15, 2+2=4", "five times three equals fifteen, two plus two equals four")
        say("😀 Great job! 👍", "Great job!")
        say("www.example.com and john@gmail.com", "example dot com and john at gmail dot com")
        say("Q3 2024 revenue up 12.5%", "Q three twenty twenty four revenue up twelve point five percent")
        say("on 25/10/2026 or 2026-10-25", "on October twenty fifth, twenty twenty six or October twenty fifth, twenty twenty six")
        say("1st, 2nd, 23rd, 101st", "first, second, twenty third, one hundred first")
        say("Dinner · 10:00 – 11:00 PM", "Dinner, ten to eleven pee em")
        say("Reminder: Dentist tomorrow at 9am", "Reminder: Dentist tomorrow at nine ay em")
        say("Your flight UA 857 departs at 21:45 from Gate B12", "Your flight UA eight hundred fifty seven departs at nine forty five pee em from Gate B twelve")
        say("Your OTP is 482913. Do not share it with anyone.", "Your OTP is four eight two, nine one three. Do not share it with anyone.")
        say("Your Amazon order #402-1234567-1234567 has shipped", "Your Amazon order number four zero two, one two three, four five six seven, one two three, four five six seven has shipped")
        say("Payment of Rs.1,250.00 received from Priya via UPI", "Payment of one thousand two hundred fifty rupees received from Priya via UPI")
        say("I think, for the most part I agree with you.", "I think, for the most part I agree with you.")
        say("Then Mary I went to the store and I said hi.", "Then Mary I went to the store and I said hi.")
        say("Henry VIII had six wives. Louis XIV built Versailles.", "Henry the eighth had six wives. Louis the fourteenth built Versailles.")
        say("King George III's madness was famous.", "King George the third's madness was famous.")
        say("Martin Luther King Jr. spoke in 1963.", "Martin Luther King Jr. spoke in nineteen sixty three.")
        say("Rocky II and Super Bowl LVIII were in Washington DC.", "Rocky two and Super Bowl fifty eight were in Washington DC.")
        say("In the 19th century, about 1500 people lived there.", "In the nineteenth century, about one thousand five hundred people lived there.")
        say("The meeting is on 10/11 at 3:30.", "The meeting is on October eleventh at three thirty.")
        say("Score was 3-2 at half time; ratio 16:9.", "Score was three to two at half time; ratio sixteen nine.")
        say("Call +1 (415) 555-0132 or +91 98765 43210.", "Call plus one, four one five, five five five, zero one three two or plus nine one, nine eight seven six five, four three two one zero.")
        say("Temperature dropped to -5°C overnight.", "Temperature dropped to minus five degrees Celsius overnight.")
        say("It costs $4.2M, or $5k per seat, or 99¢.", "It costs four point two million dollars, or five thousand dollars per seat, or ninety nine cents.")
        say("COVID-19 cases rose 12% in Q2 2021.", "COVID nineteen cases rose twelve percent in Q two twenty twenty one.")
        say("See https://en.wikipedia.org/wiki/Alice%27s_Adventures for more.", "See en dot wikipedia dot org for more.")
        say("Email me at karan.singla@whissle.ai", "Email me at karan dot singla at whissle dot ai")
        say("Follow @nasa and #SpaceX for updates.", "Follow at nasa and hashtag SpaceX for updates.")
        say("The years 1914–1918 were hard.", "The years nineteen fourteen to nineteen eighteen were hard.")
        say("Pages 23-45 cover chapter 3.", "Pages twenty three to forty five cover chapter three.")
        say("He was born on 5 March 1990 in São Tomé.", "He was born on the fifth of March nineteen ninety in Sao Tome.")
        say("I'll be there at 7:05 a.m. sharp.", "I'll be there at seven oh five ay em sharp.")
        say("Ready in 1h 30m.", "Ready in one hour thirty minutes.")
        say("\"Well,\" said Alice, \"I'm 7 and a half.\"", "\"Well,\" said Alice, \"I'm seven and a half.\"")
        say("It's 24/7, 365 days a year.", "It's twenty four seven, three hundred sixty five days a year.")
        say("Windows 11, iPhone 15 Pro, Pixel 9a, PS5", "Windows eleven, iPhone fifteen Pro, Pixel nine a, PS five")
        say("A 3.5mm jack and a 4000mAh battery.", "A three point five millimeters jack and a four thousand milliamp hours battery.")
        say("CHAPTER XIV. THE END", "CHAPTER fourteen. THE END")
        say("BOOK II", "BOOK two")
        say("Part I", "Part one")
        say("Act III, Scene ii", "Act three, Scene two")
        say("Stave I: Marley's Ghost", "Stave one: Marley's Ghost")
        say("He paid £3.50 for 2 coffees.", "He paid three pounds and fifty pence for two coffees.")
        say("The ½ cup of sugar, ¼ tsp salt", "The one half cup of sugar, one quarter teaspoon salt")
        say("1,000,000 users and 2,500,000 downloads.", "one million users and two million five hundred thousand downloads.")
        say("Account no. 1234567890 is credited.", "Account number one two three, four five six, seven eight nine zero is credited.")
        say("The year 2000 problem, Y2K.", "The year two thousand problem, Y two K.")
        say("From Mon–Fri, 9–5.", "From Monday to Friday, nine to five.")
        say("The ﬁrst ﬂoor café is at №3.", "The first floor cafe is at number three.")
        say("Добрый день, नमस्ते, こんにちは friend!", "friend!")
        say("Watch it at 1.5x speed.", "Watch it at one point five times speed.")
        say("Version v3.2 is out; see section 4.1.", "version three point two is out; see section four point one.")
        say("Born c. 1450, died 1519.", "Born circa fourteen fifty, died fifteen nineteen.")
        say("4 out of 5 doctors.", "four out of five doctors.")
        say("The 2015 people who came.", "The two thousand fifteen people who came.")
        say("It was 2,015 meters high.", "It was two thousand fifteen meters high.")
    }

    func testDayFirstDates() {
        Spoken.dayFirst = true
        XCTAssertEqual("on December third, twenty twenty six", Spoken.say("on 3/12/2026"))
        XCTAssertEqual("due March fourth", Spoken.say("due 4/3"))
    }

    func testDayFirstDefault() {
        XCTAssertFalse(Spoken.defaultDayFirst(Locale(identifier: "en_US")))
        XCTAssertTrue(Spoken.defaultDayFirst(Locale(identifier: "en_GB")))
        XCTAssertTrue(Spoken.defaultDayFirst(Locale(identifier: "hi_IN")))
    }

    func testMarkupKept() {
        XCTAssertEqual("at ten o'clock [pause 300ms] *really*", Spoken.say("at 10:00 [pause 300ms] *really*", markup: true))
    }

    func testIdempotentAndNoDigits() {
        let xs = ["Dinner · 10:00 – 11:00 PM", "CHAPTER XIV. THE END", "$1,299.99", "Call +1 (415) 555-0132.", "1990s"]
        for x in xs {
            let once = Spoken.say(x)
            XCTAssertEqual(once, Spoken.say(once))
            XCTAssertFalse(once.matches(".*\\d.*"), once)
        }
    }

    func testNumberWords() {
        XCTAssertEqual("one hundred first", Spoken.ordinal(101))
        XCTAssertEqual("twenty twenty six", Spoken.year(2026))
        XCTAssertEqual("two thousand five", Spoken.year(2005))
        XCTAssertEqual("nineteen oh five", Spoken.year(1905))
        XCTAssertEqual("nineteen hundred", Spoken.year(1900))
        XCTAssertEqual("one million two thousand three", Spoken.cardinal(1002003))
        XCTAssertEqual(1994, Spoken.romanValue("MCMXCIV"))
    }

    func testLanguagesTheVoiceCantSpeak() {
        XCTAssertEqual("Hindi", Spoken.unspoken("नमस्ते भाई, कैसे हो?"))
        XCTAssertEqual("Russian", Spoken.unspoken("Добрый день"))
        XCTAssertEqual("Japanese", Spoken.unspoken("こんにちは"))
        XCTAssertEqual(nil, Spoken.unspoken("Hi भाई"))
        XCTAssertEqual(nil, Spoken.unspoken("Café at 10:00"))
        XCTAssertEqual(nil, Spoken.unspoken("😂👍"))
    }
}
