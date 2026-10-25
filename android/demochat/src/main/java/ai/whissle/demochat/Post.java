package ai.whissle.demochat;

import android.app.Activity;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Person;
import android.content.Intent;
import android.os.Bundle;

/** Posts one demo notification from intent extras and finishes: a chat message, or a ringing call (CallStyle). */
public class Post extends Activity {
    private static final String MESSAGES = "messages", CALLS = "calls";

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        Intent i = getIntent();
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.createNotificationChannel(new NotificationChannel(MESSAGES, "Messages", NotificationManager.IMPORTANCE_HIGH));
        nm.createNotificationChannel(new NotificationChannel(CALLS, "Calls", NotificationManager.IMPORTANCE_HIGH));
        if (i.getBooleanExtra("cancel", false)) {
            nm.cancelAll();
            finish();
            return;
        }
        String from = i.getStringExtra("from") != null ? i.getStringExtra("from") : "Priya";
        Person who = new Person.Builder().setName(from).setKey(from).build();
        PendingIntent open = PendingIntent.getActivity(this, 0, new Intent(this, Post.class).putExtra("cancel", true),
                PendingIntent.FLAG_IMMUTABLE);
        Notification n;
        if (i.getBooleanExtra("call", false)) {
            n = new Notification.Builder(this, CALLS)
                    .setSmallIcon(android.R.drawable.sym_action_call)
                    .setContentTitle(from)
                    .setContentText("Incoming voice call")
                    .setCategory(Notification.CATEGORY_CALL)
                    .setFullScreenIntent(open, true)
                    .setOngoing(true)
                    .setStyle(Notification.CallStyle.forIncomingCall(who, open, open))
                    .build();
            nm.notify(2, n);
        } else {
            Person me = new Person.Builder().setName("You").build();
            Notification.MessagingStyle st = new Notification.MessagingStyle(me);
            String group = i.getStringExtra("group");
            if (group != null) st.setConversationTitle(group).setGroupConversation(true);
            st.addMessage(i.getStringExtra("text"), System.currentTimeMillis(), who);
            n = new Notification.Builder(this, MESSAGES)
                    .setSmallIcon(android.R.drawable.sym_action_chat)
                    .setStyle(st)
                    .setCategory(Notification.CATEGORY_MESSAGE)
                    .setContentIntent(open)
                    .setAutoCancel(true)
                    .build();
            nm.notify((group != null ? group : from).hashCode(), n);
        }
        finish();
    }
}
