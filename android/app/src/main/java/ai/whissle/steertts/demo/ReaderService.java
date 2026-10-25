package ai.whissle.steertts.demo;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.media.MediaMetadata;
import android.media.session.MediaSession;
import android.media.session.PlaybackState;
import android.os.Build;
import android.os.IBinder;

/**
 * Keeps reading aloud alive outside the app and puts it where media controls live: a media-playback foreground
 * service with a MediaSession (lock screen, notification shade, Bluetooth and headset buttons) and a media
 * notification with previous / play-pause / next / close. Narrator calls update() whenever its state changes.
 */
public class ReaderService extends Service implements Narrator.Listener {
    private static final String CHANNEL = "reading", ACTION = "ai.whissle.steertts.READER";
    private static final int ID = 2;
    private MediaSession session;
    private Narrator narrator;

    private static volatile boolean running;

    /**
     * Playback started: start the service in the foreground. Later changes reach a running service through its
     * Narrator listener; starting a service from the background (a pause from the lock screen) is not allowed.
     */
    static void update(Context c) {
        Narrator n = Narrator.get(c);
        if (n.doc() == null || running || !(n.playing() || n.suspended())) return;
        try {
            c.startForegroundService(new Intent(c, ReaderService.class).putExtra("fg", true));
        } catch (RuntimeException e) {   // ForegroundServiceStartNotAllowedException: the app is in the background
            android.util.Log.w(MainActivity.TAG, "reader service not started: " + e);
        }
    }

    static void stop(Context c) { c.stopService(new Intent(c, ReaderService.class)); }

    @Override public IBinder onBind(Intent i) { return null; }

    @Override public void onCreate() {
        super.onCreate();
        running = true;
        NotificationChannel ch = new NotificationChannel(CHANNEL, "Reading aloud", NotificationManager.IMPORTANCE_LOW);
        ch.setShowBadge(false);
        getSystemService(NotificationManager.class).createNotificationChannel(ch);
        narrator = Narrator.get(this);
        session = new MediaSession(this, "SteerTTS");
        session.setCallback(new MediaSession.Callback() {
            @Override public void onPlay() {
                if (narrator.doc() == null) resumeLast();
                narrator.play();
            }
            @Override public void onPause() { narrator.pause(); }
            @Override public void onSkipToNext() { narrator.next(); }
            @Override public void onSkipToPrevious() { narrator.prev(); }
            @Override public void onStop() { narrator.close(); }
            @Override public void onSeekTo(long pos) { narrator.seek((int) (pos / 1000)); }
        });
        session.setSessionActivity(PendingIntent.getActivity(this, 0, new Intent(this, ReaderActivity.class),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT));
        session.setActive(true);
        narrator.addListener(this);
    }

    @Override public int onStartCommand(Intent i, int flags, int startId) {
        String cmd = i == null ? null : i.getStringExtra("cmd");
        if (narrator.doc() == null && ("toggle".equals(cmd) || "next".equals(cmd) || "prev".equals(cmd))) resumeLast();
        if (i != null && i.getBooleanExtra("fg", false)) {
            // started with startForegroundService: always go foreground right away (even if paused or closed since),
            // or Android stops the app; changed() below steps back from the foreground if reading has stopped
            foreground(narrator.doc() != null ? notification(narrator.doc(), "", true) : placeholder());
            lastState = null;
        }
        if (cmd != null) {
            switch (cmd) {
                case "toggle": narrator.toggle(); break;
                case "next": narrator.next(); break;
                case "prev": narrator.prev(); break;
                case "close": narrator.close(); stopSelf(); return START_NOT_STICKY;
                default: break;
            }
        }
        changed();
        return START_NOT_STICKY;
    }

    @Override public void onDestroy() {
        running = false;
        narrator.removeListener(this);
        session.release();
        super.onDestroy();
    }

    private String lastState, artist, artistFor;   // what the notification shows now (rebuilt only when it changes)

    @Override public void changed() {
        Doc d = narrator.doc();
        if (d == null) {
            stopForeground(STOP_FOREGROUND_REMOVE);
            stopSelf();
            return;
        }
        boolean playing = narrator.playing(), active = playing || narrator.suspended();
        int n = Math.max(1, d.sentences.size()), at = Math.min(narrator.position(), n - 1);
        String now = d.sentences.isEmpty() ? "" : Narration.display(d.sentences.get(at));
        if (!d.id.equals(artistFor)) {
            artistFor = d.id;
            artist = "SteerTTS · " + Voices.get(this).nameOf(Style.load(this).voice, null);
        }
        session.setMetadata(new MediaMetadata.Builder()
                .putString(MediaMetadata.METADATA_KEY_TITLE, Narration.display(d.title))
                .putString(MediaMetadata.METADATA_KEY_ARTIST, artist)
                .putString(MediaMetadata.METADATA_KEY_DISPLAY_SUBTITLE, now)
                .putLong(MediaMetadata.METADATA_KEY_DURATION, n * 1000L)   // one "second" per sentence: the seek bar is the book
                .build());
        session.setPlaybackState(new PlaybackState.Builder()
                .setActions(PlaybackState.ACTION_PLAY | PlaybackState.ACTION_PAUSE | PlaybackState.ACTION_PLAY_PAUSE
                        | PlaybackState.ACTION_SKIP_TO_NEXT | PlaybackState.ACTION_SKIP_TO_PREVIOUS
                        | PlaybackState.ACTION_STOP | PlaybackState.ACTION_SEEK_TO)
                .setState(playing ? PlaybackState.STATE_PLAYING : PlaybackState.STATE_PAUSED, at * 1000L, 0f)
                .build());
        String state = d.id + "|" + playing + "|" + active;
        if (state.equals(lastState)) return;   // same document, same state: the media session above has the new line
        lastState = state;
        Notification note = notification(d, now, playing);
        if (active) {
            foreground(note);
        } else {
            stopForeground(STOP_FOREGROUND_DETACH);          // paused: the notification stays, can be swiped away
            getSystemService(NotificationManager.class).notify(ID, note);
        }
    }

    private void foreground(Notification note) {
        try {
            if (Build.VERSION.SDK_INT >= 29) startForeground(ID, note, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK);
            else startForeground(ID, note);
        } catch (RuntimeException e) {
            // Android 12+ may refuse to start a foreground service from the background (a headset or lock-screen
            // play after the app was paused for long): keep playing with a plain notification instead of crashing
            android.util.Log.w(MainActivity.TAG, "reader: foreground refused: " + e);
            try {
                getSystemService(NotificationManager.class).notify(ID, note);
            } catch (SecurityException ignored) {
            }
        }
    }

    /** A notification for the moment between a foreground start and stopping (nothing to read any more). */
    private Notification placeholder() {
        return new Notification.Builder(this, CHANNEL).setSmallIcon(R.drawable.ic_launcher_monochrome)
                .setContentTitle("SteerTTS").setContentText("Reading aloud").build();
    }

    /**
     * Play pressed on a notification, a headset or the lock screen after Android ended the app: the document read most
     * recently and not finished opens again where it was left.
     */
    private void resumeLast() {
        for (Doc.Meta m : Doc.list(this)) {
            if (m.done) continue;
            Doc d = Doc.load(this, m.id);
            if (d != null) narrator.open(d);
            return;
        }
    }

    private android.graphics.drawable.Icon icon(int res) { return android.graphics.drawable.Icon.createWithResource(this, res); }

    private PendingIntent cmd(String c, int code) {
        return PendingIntent.getService(this, code, new Intent(this, ReaderService.class).putExtra("cmd", c),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
    }

    private Notification notification(Doc d, String now, boolean playing) {
        return new Notification.Builder(this, CHANNEL)
                .setSmallIcon(R.drawable.ic_launcher_monochrome)
                .setContentTitle(Narration.display(d.title))
                .setContentText(now)
                .setContentIntent(session.getController().getSessionActivity())
                .setDeleteIntent(cmd("close", 4))
                .setOngoing(playing)
                .setVisibility(Notification.VISIBILITY_PUBLIC)
                .addAction(new Notification.Action.Builder(icon(R.drawable.ic_prev), "Previous", cmd("prev", 1)).build())
                .addAction(new Notification.Action.Builder(icon(playing ? R.drawable.ic_pause : R.drawable.ic_play),
                        playing ? "Pause" : "Play", cmd("toggle", 2)).build())
                .addAction(new Notification.Action.Builder(icon(R.drawable.ic_next), "Next", cmd("next", 3)).build())
                .setStyle(new Notification.MediaStyle().setMediaSession(session.getSessionToken()).setShowActionsInCompactView(0, 1, 2))
                .build();
    }
}
