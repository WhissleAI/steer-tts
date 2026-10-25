package ai.whissle.steertts.demo;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The voice library: every voice of the model with a human name and a plain description from its measured pitch
 * register and pace (the model's voices.json), a featured set, and the user's favourites. Names are assigned in voice
 * order from fixed lists, so a voice keeps its name across installs.
 */
final class Voices {
    static final class Voice {
        final String id, gender, pitch, pace, name;

        Voice(String id, String gender, String pitch, String pace, String name) {
            this.id = id;
            this.gender = gender;
            this.pitch = pitch;
            this.pace = pace;
            this.name = name;
        }

        /** "Deep, relaxed" from the measured register and pace. */
        String describe() {
            String p = "low".equals(pitch) ? "Deep" : "high".equals(pitch) ? "Bright" : "Warm";
            String s = "slow".equals(pace) ? "relaxed" : "fast".equals(pace) ? "brisk" : "steady";
            return p + ", " + s;
        }

        String gender() { return "female".equals(gender) ? "Female" : "Male"; }
    }

    // featured: both defaults, then a spread of registers and paces
    static final String[] FEATURED = {"voice_1688", "voice_2500", "voice_0010", "voice_1469", "voice_0185", "voice_0413",
            "voice_1226", "voice_0213"};

    private static final String[] FEMALE = {"Ava", "Mia", "Zoe", "Lily", "Emma", "Nora", "Ruby", "Iris", "Maya", "Leah",
            "Clara", "Ella", "Grace", "Hazel", "Ivy", "Jade", "Kate", "Lena", "Luna", "Mila", "Nina", "Olive", "Paige",
            "Quinn", "Rose", "Sara", "Tessa", "Vera", "Willow", "Yara", "Abby", "Beth", "Cora", "Dana", "Eva", "Faith",
            "Gemma", "Hope", "Isla", "June", "Kira", "Lucy", "Molly", "Nadia", "Opal", "Piper", "Rhea", "Sofia", "Tara",
            "Una", "Violet", "Wren", "Alice", "Brooke", "Chloe", "Daisy", "Eden", "Fiona", "Gwen", "Holly", "Ingrid",
            "Joy", "Kylie", "Laura", "Megan", "Naomi", "Olivia", "Penny", "Riley", "Stella", "Tina", "Uma", "Val",
            "Wendy", "Amber", "Bella", "Celia", "Diana", "Elise", "Freya", "Greta", "Heidi", "Irene", "Jenna", "Kelly",
            "Lara", "Martha", "Nell", "Orla", "Phoebe", "Rachel", "Sienna", "Thea", "Ursula", "Vivian", "Zara",
            "Anna", "Bonnie", "Carmen", "Della", "Erin", "Flora", "Gloria", "Hannah", "Isabel"};
    private static final String[] MALE = {"Leo", "Max", "Sam", "Noah", "Eli", "Owen", "Jack", "Theo", "Finn", "Luke",
            "Adam", "Ben", "Cole", "Dan", "Evan", "Felix", "Gus", "Hugo", "Ian", "Jude", "Kai", "Liam", "Milo", "Nate",
            "Oscar", "Paul", "Reed", "Seth", "Tom", "Wade", "Alex", "Blake", "Carl", "Dean", "Ezra", "Frank", "Grant",
            "Hank", "Isaac", "Jake", "Kyle", "Lance", "Mark", "Neil", "Otto", "Pete", "Ray", "Scott", "Troy", "Vince",
            "Will", "Aaron", "Brian", "Chris", "Derek", "Eric", "Fred", "Gary", "Henry", "Ivan", "Jason", "Kevin",
            "Leon", "Mason", "Nick", "Omar", "Peter", "Ross", "Simon", "Tyler", "Victor", "Wes", "Andy", "Bruce",
            "Caleb", "David", "Edward", "Floyd", "George", "Harry", "Jasper", "Keith", "Louis", "Martin", "Nolan",
            "Oliver", "Philip", "Ralph", "Steve", "Tony", "Walter", "Arthur", "Bryce", "Connor", "Dylan", "Elliot",
            "Gavin", "Harvey", "Jonah", "Kurt", "Logan", "Miles", "Nigel", "Patrick", "Roman", "Silas", "Toby"};

    private static Voices instance;
    private final Map<String, Voice> byId = new LinkedHashMap<>();
    private final SharedPreferences prefs;

    static synchronized Voices get(Context c) {
        if (instance == null || instance.byId.isEmpty()) instance = new Voices(c.getApplicationContext());   // retry a failed load
        return instance;
    }

    private Voices(Context c) {
        prefs = c.getSharedPreferences("voices", Context.MODE_PRIVATE);
        try {
            File f = new File(SteerTTS.installModel(c), "voices.json");
            JSONArray a;
            try (FileInputStream in = new FileInputStream(f)) {
                a = new JSONObject(new String(in.readAllBytes())).getJSONArray("voices");
            }
            int nf = 0, nm = 0;
            for (int i = 0; i < a.length(); ++i) {
                JSONObject v = a.getJSONObject(i);
                String g = v.optString("gender", "female");
                String name = "female".equals(g) ? FEMALE[nf++ % FEMALE.length] : MALE[nm++ % MALE.length];
                String id = v.getString("name");
                byId.put(id, new Voice(id, g, v.optString("pitch_register", "mid"), v.optString("pace", "average"), name));
            }
        } catch (Exception e) {
            android.util.Log.e("SteerTTS", "voices", e);
        }
    }

    List<Voice> all() { return new ArrayList<>(byId.values()); }

    Voice voice(String id) { return byId.get(id); }

    /** A voice id, "female" / "male" (the defaults) or null, as the name people see ("Ava"). */
    String nameOf(String id, SteerTTS e) {
        if (id == null || "female".equals(id)) id = FEATURED[0];
        else if ("male".equals(id)) id = FEATURED[1];
        Voice v = byId.get(id);
        return v != null ? v.name : id;
    }

    List<Voice> featured() {
        List<Voice> out = new ArrayList<>();
        for (String id : FEATURED) if (byId.containsKey(id)) out.add(byId.get(id));
        return out;
    }

    Set<String> favourites() { return new HashSet<>(prefs.getStringSet("fav", Collections.emptySet())); }

    boolean isFavourite(String id) { return favourites().contains(id); }

    void toggleFavourite(String id) {
        Set<String> s = favourites();
        if (!s.remove(id)) s.add(id);
        prefs.edit().putStringSet("fav", s).apply();
    }

    static List<String> featuredIds() { return Arrays.asList(FEATURED); }

    /**
     * A voice of their own for a person whose messages are read out: always the same for the same name, one of the
     * featured voices, never the user's own (so the assistant's framing and the message are told apart).
     */
    static String forPerson(String name, String own) {
        String ownId = own == null || "female".equals(own) ? FEATURED[0] : "male".equals(own) ? FEATURED[1] : own;
        List<String> pool = new ArrayList<>();
        for (String id : FEATURED) if (!id.equals(ownId)) pool.add(id);
        String key = name.trim().toLowerCase(java.util.Locale.ROOT);
        int h = 0;
        for (int i = 0; i < key.length(); ++i) h = 31 * h + key.charAt(i);   // String.hashCode, spelled out: stable
        return pool.get(Math.floorMod(h, pool.size()));
    }
}
