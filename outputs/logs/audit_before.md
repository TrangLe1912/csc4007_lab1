# Audit BEFORE preprocessing

## Schema / Missingness
- **n_rows**: 50000
- **columns**: ['id', 'text', 'label', 'split_orig']
- **missing_text_count**: 0
- **empty_text_count**: 0
- **missing_label_count**: 0
- **n_labels**: 2
- **label_counts**: {0: 25000, 1: 25000}

## HTML tags / Entities
- **contains_br_tag_count**: 29200
- **contains_any_html_tag_count**: 29202
- **contains_html_entity_count**: 11
- **example_snippets_with_html**: ['I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it\'s not shot like some cheaply made porno. While my countrymen mind find it shocking, in reality sex and nudity are a major staple in Swedish cinema. Even Ingmar Bergman, arguably their answer to good old boy John Ford, had sex scenes in his films.<br /><br />I do commend the filmmakers for the fact that any sex shown in the film is shown for artistic purposes rather than just to shock people and make money to be shown in pornographic theaters in America. I AM CURIOUS-YELLOW is a good film for anyone wanting to study the meat and potatoes (no pun intended) of Swedish cinema. But really, this film doesn\'t have much of a plot.', "If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />One might feel virtuous for sitting thru it because it touches on so many IMPORTANT issues but it does so without any discernable motive. The viewer comes away with no new perspectives (unless one comes up with one while one's mind wanders, as it will invariably do during this pointless film).<br /><br />One might better spend one's time staring out a window at a tree growing.<br /><br />", "This film was probably inspired by Godard's Masculin, féminin and I urge you to see that film instead.<br /><br />The film has two strong elements and those are, (1) the realistic acting (2) the impressive, undeservedly good, photo. Apart from that, what strikes me most is the endless stream of silliness. Lena Nyman has to be most annoying actress in the world. She acts so stupid and with all the nudity in this film,...it's unattractive. Comparing to Godard's film, intellectuality has been replaced with stupidity. Without going too far on this subject, I would say that follows from the difference in ideals between the French and the Swedish society.<br /><br />A movie of its time, and place. 2/10."]

## Distribution / Length
- **label_counts**: {0: 25000, 1: 25000}
- **imbalance_ratio_max_over_min**: 1.0
- **len_chars_min**: 32
- **len_chars_median**: 970
- **len_chars_p95**: 3391
- **len_chars_max**: 13704
- **median_len_by_label**: {'0': 973.0, '1': 968.0}
- **note**: Length bias is a risk signal; do not conclude shortcut without further tests.

## Duplicates / Near-duplicates
- **exact_dup_count**: 824
- **exact_dup_ratio**: 0.01648
- **label_inconsistent_dup_groups**: 0
- **near_dup_pairs_found_in_sample**: 0
- **near_dup_examples**: []

## Leakage demo (TF-IDF fit all vs train-only)
- **demo_note**: Educational demo: fitting preprocessing on all data before split causes contamination.
- **vocab_size_bad_fit_all**: 50000
- **vocab_size_good_fit_train**: 50000
- **idf_bad_mean**: 7.3001093507430665
- **idf_good_mean**: 7.296006161010017
- **fix**: Split first. Fit preprocessing on train only; transform val/test.

