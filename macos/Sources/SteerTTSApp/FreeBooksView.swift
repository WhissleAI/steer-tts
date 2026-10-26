import SwiftUI

/// Free books from Project Gutenberg: shelves and search, most read first; a book you already have says so and opens
/// your copy where you left it.
struct FreeBooksView: View {
    @EnvironmentObject var app: AppModel
    @EnvironmentObject var library: Library
    @State private var shelf = 1                 // children's stories first, as on Android
    @State private var query = ""
    @State private var searched: String?
    @State private var books: [Gutenberg.Book] = []
    @State private var loading = false
    @State private var more = true
    @State private var error: String?
    @State private var nextStart = 1           // OPDS pages are 25 entries, whether or not we show them all

    private var current: String { searched ?? Gutenberg.shelves[shelf].query }

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 10) {
                Text("70,000 books, free to read and hear").font(.title2.weight(.semibold))
                Text("Public-domain classics from Project Gutenberg. Stories are performed: each character gets a voice of their own.")
                    .foregroundStyle(.secondary)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(Gutenberg.shelves.indices, id: \.self) { i in
                            Button(Gutenberg.shelves[i].name) { searched = nil; query = ""; shelf = i; load(reset: true) }
                                .buttonStyle(Chip(on: searched == nil && shelf == i))
                        }
                    }
                }
            }
            .padding(.horizontal, 24).padding(.top, 20).padding(.bottom, 12)
            Divider()
            List {
                ForEach(books) { b in
                    let have = library.docs.contains { $0.source == b.source || ($0.source.isEmpty && $0.displayTitle.hasPrefix(b.title)) }
                    HStack(spacing: 12) {
                        Image(systemName: "book.closed").foregroundStyle(Theme.accent)
                            .frame(width: 30, height: 30).background(Theme.accentSoft, in: RoundedRectangle(cornerRadius: 7))
                        VStack(alignment: .leading, spacing: 2) {
                            Text(b.title).font(.body.weight(.medium)).lineLimit(2)
                            Text([b.author, have ? "In your list" : ""].filter { !$0.isEmpty }.joined(separator: " · "))
                                .font(.callout).foregroundStyle(have ? Theme.accent : .secondary)
                        }
                        Spacer()
                        Button(have ? "Open" : "Listen") { app.open(book: b) }
                    }
                    .padding(.vertical, 3)
                    .contentShape(Rectangle())
                    .onTapGesture(count: 2) { app.open(book: b) }
                }
                if more && !books.isEmpty {
                    HStack { Spacer(); if loading { ProgressView().controlSize(.small) } else { Button("More Books") { load(reset: false) } }; Spacer() }
                        .onAppear { if !loading { load(reset: false) } }
                }
            }
            .listStyle(.inset)
            .overlay {
                if loading && books.isEmpty { ProgressView() }
                else if let e = error, books.isEmpty {
                    VStack(spacing: 10) {
                        Text(e).foregroundStyle(.secondary)
                        Button("Try Again") { load(reset: true) }
                    }
                } else if books.isEmpty && !loading {
                    Text("No books found.").foregroundStyle(.secondary)
                }
            }
        }
        .searchable(text: $query, prompt: "Title or author")
        .onSubmit(of: .search) {
            let q = query.trimmingCharacters(in: .whitespaces)
            searched = q.isEmpty ? nil : q
            load(reset: true)
        }
        .task { if books.isEmpty { load(reset: true) } }
    }

    private func load(reset: Bool) {
        let q = current
        let start = reset ? 1 : nextStart
        if reset { books = []; more = true }
        loading = true
        error = nil
        Task {
            do {
                let page = try await Gutenberg.page(q, start: start)
                guard q == current else { return }
                let have = Set(books.map(\.id))
                books += page.filter { !have.contains($0.id) }
                more = !page.isEmpty
                nextStart = start + 25
            } catch {
                self.error = "Project Gutenberg could not be reached. Check your connection."
            }
            loading = false
        }
    }
}
